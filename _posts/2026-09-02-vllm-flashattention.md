---
layout: post
title: "FlashAttention：从 IO-aware tiling 到 Hopper 异步流水"
date: 2026-09-02 10:00 +0800
tags: [vLLM, LLM 推理, FlashAttention, GPU, FP8, Attention]
toc: true
math: true
permalink: /vllm/flashattention/
---

## 本篇要回答什么

标准 attention 的 FLOPs 很多，但在 GPU 上还可能更受数据搬运限制。FlashAttention v1 怎样在不保存 $N\times N$ 中间矩阵的情况下得到精确 softmax？v2 为什么改变 thread block 和 warp 分工？v3 又如何利用 Hopper 的 TMA、WGMMA 和 FP8？

## 1. 标准 attention 慢在哪里

$$
S=\frac{QK^T}{\sqrt d},\qquad
P=\operatorname{softmax}(S),\qquad
O=PV.
$$

若序列长度为 $N$，$S$、$P$ 都是 $N\times N$。传统分离实现可能经历：

```text
读 Q/K → 计算 S → S 写 HBM
读 S   → softmax → P 写 HBM
读 P/V → 计算 O → O 写 HBM
```

HBM 容量大，但数据往返相对片上 SRAM/register 昂贵。FlashAttention 没有把完整 attention 改成稀疏或低秩近似；它通过 tiling（分块）、online softmax 和 kernel fusion，减少 $N^2$ 中间结果的 HBM 读写。

FLOPs 是完成一次任务需要多少浮点运算；FLOPS 是硬件每秒能完成多少浮点运算。一个 kernel FLOPs 很高，不代表它能达到峰值 FLOPS：若每做少量运算就要等 HBM，计算单元仍会空闲。

GPU 存储层级可用一个不带具体芯片数值的心智图理解：

```text
HBM：容量最大、跨 SM 可见、延迟/能耗较高
  ↕ cooperative/asynchronous copies
shared memory / SRAM：每个 CTA/SM 的片上 tile staging
  ↕ load/store
register：每线程/warp 的标量与小 fragment，最快但容量最紧
```

普通分离 attention 对 $S/P\in\mathbb{R}^{B\times H_q\times N\times N}$ 的写回/重读会随 $N^2$ 增长。例如 $B=1,H_q=32,N=8192$、每元素 2 bytes，单张 $N\times N$ 张量就是 4 GiB；先写 S 再写 P 会制造巨大的中间流量。实际 kernel 可能融合某些阶段，这个数只说明为何物化矩阵昂贵。

## 2. FlashAttention v1：tile 在片上完成生命周期

将 Q、K、V 切成小块。对一个 $Q_i$ 与一个 $K_j,V_j$ tile：

```text
载入 Q_i、K_j、V_j
  → 计算局部 S_ij = Q_i K_j^T / sqrt(d)
  → 更新该 Q 行的 softmax 统计量
  → 立即完成局部 P_ij V_j 累加
  → 丢弃 S_ij 和 P_ij
```

softmax 的分母需要整行，单个 tile 却看不到未来 keys，这由 online softmax 解决。

naive 与 tiled 教学伪代码对照：

```python
# naive：概念上物化 N×N scores/probabilities
S = Q @ K.T / sqrt(d)
P = softmax(S, axis=-1)
O = P @ V

# tiled：对每个 Q block扫描 K/V blocks
for Qi in q_tiles(Q):
    m = -inf; l = 0; A = 0
    for Kj, Vj in kv_tiles(K, V):
        Sij = Qi @ Kj.T / sqrt(d)
        m, l, A = online_update(m, l, A, Sij, Vj)
    write(Oi=A/l)
```

两者输入/输出 shape 都是 Q/K/V `[N,d]` 到 O `[N,d]`；差别在中间状态的生命周期。tiled 版本只让当前 `Sij/Pij` 活在片上，长期状态是每个 query row 的 $m,l,A$。它没有减少所有 $(i,j)$ 点积，只减少 HBM 中间存取。

## 3. Online Softmax

对每个 query 行维护当前最大值 $m$、指数和 $l$ 与未归一化输出累加 $A$。初始为

$$
m=-\infty,\qquad l=0,\qquad A=0.
$$

读入新 score block $S_b$ 后：

$$
m'=\max\left(m,\operatorname{rowmax}(S_b)\right),
$$

$$
l'=e^{m-m'}l+\operatorname{rowsum}(e^{S_b-m'}),
$$

$$
A'=e^{m-m'}A+e^{S_b-m'}V_b.
$$

处理完全部 K/V blocks 后得到 $O=A/l$。最大值变化时，$e^{m-m'}$ 把旧累加量换到新的数值基准，因此不必重新访问此前 scores。结果仍对应完整 softmax attention；“exact”表示没有稀疏/低秩算法近似，不表示浮点运算没有舍入差异。

为什么旧累加必须重缩放？旧和保存的是 $\sum_{j\in old}e^{s_j-m}$；新基准变为 $m'$ 后，同一项应写成 $e^{s_j-m'}=e^{s_j-m}e^{m-m'}$。漏掉该因子会把旧块放在错误尺度上，甚至让较早的小最大值获得过大权重。

### 可执行数值对照

```python
from math import exp, inf

def weighted_sum(weights, values):
    return [sum(w*row[j] for w, row in zip(weights, values))
            for j in range(len(values[0]))]

def ordinary(scores, values):
    m = max(scores)
    weights = [exp(s-m) for s in scores]
    z = sum(weights)
    return [x/z for x in weighted_sum(weights, values)]

def online(scores, values, block=2):
    m, l = -inf, 0.0
    a = [0.0] * len(values[0])
    for start in range(0, len(scores), block):
        s = scores[start:start+block]
        v = values[start:start+block]
        m_new = max(m, max(s))
        old_scale = 0.0 if m == -inf else exp(m-m_new)
        weights = [exp(x-m_new) for x in s]
        local = weighted_sum(weights, v)
        a = [old_scale*x+y for x, y in zip(a, local)]
        l = old_scale*l + sum(weights)
        m = m_new
    return [x/l for x in a]

s = [1000.0, 999.0, 1002.0, 998.0]
v = [[float(3*i+j) for j in range(3)] for i in range(4)]
expected, got = ordinary(s, v), online(s, v)
print(expected, got)
assert max(abs(a-b) for a, b in zip(got, expected)) < 1e-12
```

输入是一行 4 个 scores 和 V `[4,3]`，输出是一行 O `[3]`。状态 `m/l/a` 分别是标量、标量、`[3]`。代码验证 block online 公式等价于稳定 softmax；它没有 causal mask、batch/head 维度、GPU tiling 或有限精度优化，不能测 FlashAttention 性能。

## 4. 反向为何不用长期保存每个 $i,j$

训练反向确实需要局部 $P_{ij}$ 出现，但“需要计算”不等于“必须同时长期存储”。前向通常保存 Q、K、V、输出 O 和每个 query 行的 log-sum-exp：

$$
L_i=\log\sum_j e^{S_{ij}}.
$$

反向处理一个 tile 时重算

$$
S_{ij}=Q_iK_j^T/\sqrt d,
\qquad P_{ij}=e^{S_{ij}-L_i}.
$$

Softmax 梯度中的行归约还可利用

$$
D_i=\sum_jP_{ij}dP_{ij}=dO_i\cdot O_i.
$$

随后在片上计算并累加：

$$
dP_{ij}=dO_iV_j^T,
\qquad dS_{ij}=P_{ij}\odot(dP_{ij}-D_i),
$$

$$
dQ_i\mathrel{+}=dS_{ij}K_j,
\qquad dK_j\mathrel{+}=dS_{ij}^TQ_i,
\qquad dV_j\mathrel{+}=P_{ij}^TdO_i.
$$

局部 $S/P/dP/dS$ 用完即覆盖。长期保存从 $O(N^2)$ 中间矩阵降到约 $O(Nd)+O(N)$，代价是反向重计算。vLLM 推理主要走 forward path，但这一区分解释了 FlashAttention 论文为何同时讨论训练显存。

$D_i=dO_i\cdot O_i$ 的推导来自：

$$
O_i=\sum_jP_{ij}V_j,\qquad
dP_{ij}=dO_i\cdot V_j,
$$

所以

$$
\sum_jP_{ij}dP_{ij}
=\sum_jP_{ij}(dO_i\cdot V_j)
=dO_i\cdot\sum_jP_{ij}V_j
=dO_i\cdot O_i.
$$

它把 softmax backward 每行需要的归约量压成由已保存 $O_i$ 与上游梯度 $dO_i$ 可直接得到的标量。

```python
# 教学伪代码：tiled backward
D = row_sum(dO * O)                       # [N]
zero(dQ, dK, dV)
for Kj, Vj in kv_tiles(K, V):
    local_dK = 0; local_dV = 0
    for Qi, dOi, Li, Di in q_tiles(Q, dO, L, D):
        Sij = Qi @ Kj.T / sqrt(d)          # 重算，不从 HBM 读 N×N
        Pij = exp(Sij - Li[:, None])
        dPij = dOi @ Vj.T
        dSij = Pij * (dPij - Di[:, None])
        dQi += dSij @ Kj / sqrt(d)
        local_dK += dSij.T @ Qi / sqrt(d)
        local_dV += Pij.T @ dOi
    accumulate(dKj, local_dK)
    accumulate(dVj, local_dV)
```

输入 Q/K/V/O/L 和 dO；输出 dQ/dK/dV，shape 与原张量一致。长期保存的 L 是 `[N]` 而非 P `[N,N]`。真实实现还要处理 causal mask、dropout、head/batch、并行归约、原子更新和布局；伪代码只展示“重算 tile、立即消费”。

## 5. FlashAttention v2：让更多 SM 工作

v1 已减少 IO，却仍可能因工作划分吃不满 GPU。长上下文常迫使单卡 batch size 变小；若并行任务只有 `batch × heads`，数量可能少于 Streaming Multiprocessors（SMs）。

v2 保留 tiling 和 online softmax，重点改进：

1. 沿 Q 序列维并行：不同 Q blocks 交给不同 thread blocks；
2. 减少非 matmul FLOPs：中间维护未归一化累加，推迟部分除法和缩放；
3. 使用 sliced-Q：warps 共享 K/V，各自处理不同 Q rows，避免同一输出的跨 warp 归约。

```text
v1 sliced-K（概念图）
warp 0: Q × K0 → O 的一部分
warp 1: Q × K1 → O 的一部分
                  ↓ shared-memory reduction

v2 sliced-Q
warp 0: Q0 × K → O0
warp 1: Q1 × K → O1
                  输出不重叠，减少同步与归约
```

所以 v1 主要解决中间矩阵 IO，v2 进一步解决并行粒度、warp 通信和 GPU 利用率。

v1 当然也并行处理 Q/K/V：不同 batch/head 可给不同 thread blocks，一个 block 内多个 warps 共同处理 tiles。问题不是“完全没有并行”，而是并行网格和 warp 分工不够理想：batch×heads 太小时 CTA 数不足，sliced-K 又让 warps 对同一 Q 输出做部分和，随后需要 shared-memory 归约。v2 让 Q sequence blocks 也成为独立并行维，并让不同 warps 拥有不同 Q rows。

```text
v2 概念算法图
Q blocks: Q0  Q1  Q2  Q3
            │   │   │   │      可分派给不同 CTAs/SMs
K/V tiles: K0V0 → K1V1 → K2V2 → ...
            │       每个 Q block 独立维护 m,l,A
            └────── online softmax ──────→ O0/O1/O2/O3
```

sequence parallelism 在这里指把同一 attention head 的 query sequence 切成多个 Q blocks，让 `batch × heads × q_blocks` 提供更多 CTAs。它不是训练语境中唯一的“序列并行”定义。若 batch=1、heads=8，而 GPU 有数十/上百个 SM，只按 batch×heads 分配最多 8 个大任务会闲置大量 SM；再沿 Q 维切块才能提升占用。

“减少 non-matmul FLOPs”并非说 exp/max/scale 不重要，而是 Tensor Cores 的 matmul 吞吐远高于普通标量运算；过多 rescale、除法、shared-memory reduction 会阻断高吞吐矩阵乘。v2 调整算法排列，减少这些工作和跨 warp 通信。

## 6. MEA、近似 attention 与 FlashAttention

有些文章把 Memory-Efficient Attention 写成 EMA；更常见的是 MEA。这里不是训练参数的 Exponential Moving Average。

| 方法 | 核心做法 | 是否减少关系数 | 数学结果 |
|---|---|---:|---|
| Memory-Efficient Attention | 分块、online softmax、重计算 | 否 | 完整 attention |
| FlashAttention | IO-aware tiling 与融合 GPU kernel | 否 | 完整 attention |
| Longformer | 局部窗口和少量全局连接 | 是 | 稀疏 attention |
| Performer | 随机特征近似 softmax kernel | 是 | 近似 attention |

Longformer 少算连接，Performer 换成近似公式，FlashAttention 基本仍计算所有有效关系，但避免把巨大中间表格反复搬进搬出 HBM。

Performer 用 FAVOR+ 随机正特征映射近似 softmax kernel：

$$
e^{q^Tk}\approx\phi(q)^T\phi(k),
$$

于是利用结合律先聚合 $\sum_j\phi(k_j)v_j^T$，避免显式形成全部 pair。随机特征数控制速度/误差；正特征帮助稳定，正交随机特征可降低方差，但仍是算法近似。Longformer 则显式规定局部窗口与少量 global tokens，是稀疏连接；MEA/FlashAttention 保持完整连接集合，是 exact family。

| 维度 | FlashAttention/MEA | Longformer | Performer/FAVOR+ |
|---|---|---|---|
| 连接 | 全部有效 $(i,j)$ | 窗口 + global | 隐式全局 |
| 主要复杂度 | 仍约 $N^2d$ FLOPs | 随窗口近线性 | 随特征数近线性 |
| 误差来源 | 浮点重排 | 改变 attention pattern | 随机特征近似 |
| 主要目的 | 降 IO/显存中间量 | 少算连接 | 核近似少算 pair |

## 7. FlashAttention v3：映射到 Hopper

FlashAttention-3 面向 Hopper（如 H100/H800），让数据搬运、Tensor Core matmul 与 softmax 更充分重叠：

- TMA（Tensor Memory Accelerator）异步搬运 tiles；
- WGMMA（Warpgroup Matrix Multiply-Accumulate）执行矩阵乘；
- warp specialization 将 producer 与 consumer 分工；
- ping-pong consumer groups 交错 matmul 和 softmax；
- 细粒度 barrier 与寄存器管理减少全 CTA 阶段同步。

```text
producer warp group:  TMA 预取下一批 Q/K/V tiles
                            │ async barrier
consumer group A:     WGMMA → softmax/accumulate
consumer group B:             WGMMA → softmax/accumulate
                       两组 ping-pong 隐藏非 matmul 阶段
```

这些优化依赖 Hopper 执行模型，不能假定旧架构会获得同样收益。

更细的双缓冲时间线：

```text
time →      t0           t1              t2              t3
producer    TMA tile 0   TMA tile 1      TMA tile 2      ...
barrier     signal 0     signal 1        signal 2
consumer A               WGMMA tile 0    softmax/A update WGMMA tile 2
consumer B                               WGMMA tile 1    softmax/B update
```

producer 发起异步 TMA 后通过 barrier 宣告 tile 可用；consumer warp group 用 WGMMA 消费。ping-pong 的意义是让一组 consumer 进行 softmax/归一化等非 matmul 工作时，另一组尽量继续矩阵乘。真实依赖与寄存器所有权比图复杂，barrier 仍用于保证“不能在数据到达前消费”和“不能在仍使用时覆盖 buffer”，不是删除同步。

v2 中同一 warp/warp group 往往同时承担搬运、matmul 和 softmax阶段，需要为多阶段 live values 预留寄存器，并在每轮切换职责。v3 specialization 让 producer 主要保留 TMA 描述符/地址状态，consumer 主要保留 WGMMA accumulator 与 softmax 状态；职责分离缩短部分变量的 live range，并减少所有 warps 都执行相同控制指令的调度开销。它不是让寄存器总数自动变少：consumer accumulator 仍很大，实际收益来自更可控的寄存器分配、异步 barrier 与阶段重叠。

## 8. FP8：高吞吐乘法与精度管理

直接把 Q/K/V 转为 FP8 容易受 outliers 控制 scale。FA3 使用两类互补手段：

- block quantization：不同块使用独立 scale，把异常值影响限制在局部；
- incoherent processing：用 Hadamard 一类正交变换打散集中在少数维度的异常值。

若 $R$ 为正交矩阵，同时旋转 Q、K：

$$
Q'=QR,\qquad K'=KR,
$$

则在没有量化误差时

$$
Q'K'^T=QRR^TK^T=QK^T.
$$

旋转只是换坐标系；额外近似来自随后的 FP8 量化。更均匀的坐标分布通常让 block scale 更有效。

FP8 attention 也不表示全部步骤均为 FP8：

```text
FP8 Q × FP8 K → 高精度 accumulator
               → 高精度 max / exp / sum
               → P tile 必要时在寄存器中量化
FP8 P × FP8 V → 高精度 accumulator
               → 最终写回时 cast 到 FP16/BF16
```

Hadamard 矩阵经 $1/\sqrt d$ 归一化后是正交矩阵 $R$，满足 $RR^T=I$。同时右乘 Q/K 保持点积；只旋转其中一个则一般不保持。变换的目标是把单维 outlier 能量摊到多个坐标，使 block scale 不被极端值完全支配，而不是凭空消除能量。

| 阶段 | 推荐精度思路 | 原因 |
|---|---|---|
| Q/K/V storage/matmul input | FP8 + per-block scale | 降带宽并用 FP8 Tensor Core |
| QK accumulator | FP16/FP32 类高精度 | 多项累加避免快速失真 |
| max/exp/sum/LSE | 高精度 | softmax 对范围敏感 |
| P tile | 可在寄存器内受控量化 | 送入 PV，不写回大矩阵 |
| PV accumulator/output | 高精度累加，末端 cast | 控制累积误差 |

这是一条数值设计原则，不代表每个 vLLM backend 都采用同一格式组合。

## 9. FP8 转换不是免费操作

最差实现会先把 FP16 Q/K/V 写入 HBM，再启动量化 kernel 读出、转 FP8、写回，attention 再读取；输出又经独立反量化 kernel。额外 launch 和 HBM 往返可能吃掉收益。

理想做法是在算子边界融合：

- QKV projection epilogue 在 accumulator 仍位于寄存器时应用 scale/cast，直接写 FP8；
- P tile 在 FlashAttention kernel 的寄存器中量化并立即送入 $PV$；
- 高精度输出 accumulator 在最终写回时转换。

转换指令不会消失，但可减少额外 HBM 往返和独立 kernel launch。长序列 prefill 的 attention matmul 为 $O(N^2d)$，QKV 转换约为 $O(Nd)$，更容易摊薄转换；单-token decode 更受 KV 带宽和固定开销影响，不能只看峰值 Tensor Core 吞吐。

完整数据流可画为：

```text
BF16 hidden
 → QKV projection accumulator（较高精度）
 → epilogue: scale + FP8 cast
 → FP8 Q/K/V tiles
 → QK WGMMA，高精度累加
 → softmax/LSE（高精度）
 → P tile 寄存器内可选 FP8 cast
 → PV WGMMA，高精度 output accumulator
 → final cast/write BF16/FP16 output
```

prefill 的收益主要可能来自大 attention matmul 的 FP8 Tensor Core 吞吐和中间 IO；decode 的 query 只有一个/少数位置，历史 KV 读取与 dequantization、固定 launch 更突出。二者必须分开 benchmark。

## 10. vLLM 中还要经过 backend 选择

FlashAttention 是算法/实现族，不等于 vLLM 的整个 attention 子系统。backend 还需满足 GPU capability、dtype、head size、block size、attention type、KV layout、quantization 和 graph 等契约。

```text
backend 已注册 → 候选存在
selector 返回 → 当前配置选择
启动日志/trace → 本次进程实际走过
profiler → 该 kernel 的时间占比与瓶颈
```

固定 v0.20.0 阅读链应包括：模型层产生/reshape QKV → 统一 `Attention` layer → `v1/attention/backend.py` 的接口契约 → `v1/attention/backends/registry.py` 的候选注册/解析 → 平台/配置 selector → 具体 backend 的 metadata builder 与 forward。检查 head size、dtype、KV cache dtype、block size、attention type、GPU capability 和 CUDA Graph 支持后，才知道候选是否适用。

证据强度依次为：源码中存在类 < selector 日志显示选中 < trace 显示本次 forward 执行 < profiler 显示 kernel 和耗时。任何一层都不能替代后一层。

## 常见误区

- “FlashAttention 将复杂度从二次变线性”：完整 attention 的主要 FLOPs 仍约为 $O(N^2d)$。
- “重计算必然更慢”：多做片上计算可能比保存和重读 $N^2$ 中间量更便宜。
- “v2 使用不同数学公式”：v2 主要重新安排并行和 warp 工作。
- “FP8 路径仍完全 exact”：它保持完整连接结构，但额外引入量化误差。

## 本篇总结

FlashAttention 的演进可压缩成三句话：v1 让 score/probability tiles 在片上完成生命周期；v2 重新分配 thread block 与 warp 工作以提高并行效率；v3 利用 Hopper 异步搬运、WGMMA、warp specialization 与受控 FP8，让数据移动、matmul 和 softmax 更充分重叠。

---

[上一篇：量化与推测解码]({{ '/vllm/quantization-speculative-decoding/' | relative_url }}) · [系列首页]({{ '/vllm/' | relative_url }})

## 资料

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Performer](https://arxiv.org/abs/2009.14794) 与 [Longformer](https://arxiv.org/abs/2004.05150)
- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
