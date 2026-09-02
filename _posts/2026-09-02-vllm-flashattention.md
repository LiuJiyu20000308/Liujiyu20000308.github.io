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

## 6. MEA、近似 attention 与 FlashAttention

有些文章把 Memory-Efficient Attention 写成 EMA；更常见的是 MEA。这里不是训练参数的 Exponential Moving Average。

| 方法 | 核心做法 | 是否减少关系数 | 数学结果 |
|---|---|---:|---|
| Memory-Efficient Attention | 分块、online softmax、重计算 | 否 | 完整 attention |
| FlashAttention | IO-aware tiling 与融合 GPU kernel | 否 | 完整 attention |
| Longformer | 局部窗口和少量全局连接 | 是 | 稀疏 attention |
| Performer | 随机特征近似 softmax kernel | 是 | 近似 attention |

Longformer 少算连接，Performer 换成近似公式，FlashAttention 基本仍计算所有有效关系，但避免把巨大中间表格反复搬进搬出 HBM。

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

## 9. FP8 转换不是免费操作

最差实现会先把 FP16 Q/K/V 写入 HBM，再启动量化 kernel 读出、转 FP8、写回，attention 再读取；输出又经独立反量化 kernel。额外 launch 和 HBM 往返可能吃掉收益。

理想做法是在算子边界融合：

- QKV projection epilogue 在 accumulator 仍位于寄存器时应用 scale/cast，直接写 FP8；
- P tile 在 FlashAttention kernel 的寄存器中量化并立即送入 $PV$；
- 高精度输出 accumulator 在最终写回时转换。

转换指令不会消失，但可减少额外 HBM 往返和独立 kernel launch。长序列 prefill 的 attention matmul 为 $O(N^2d)$，QKV 转换约为 $O(Nd)$，更容易摊薄转换；单-token decode 更受 KV 带宽和固定开销影响，不能只看峰值 Tensor Core 吞吐。

## 10. vLLM 中还要经过 backend 选择

FlashAttention 是算法/实现族，不等于 vLLM 的整个 attention 子系统。backend 还需满足 GPU capability、dtype、head size、block size、attention type、KV layout、quantization 和 graph 等契约。

```text
backend 已注册 → 候选存在
selector 返回 → 当前配置选择
启动日志/trace → 本次进程实际走过
profiler → 该 kernel 的时间占比与瓶颈
```

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
