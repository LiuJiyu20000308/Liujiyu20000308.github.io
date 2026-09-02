---
layout: post
title: "vLLM KV Cache：从历史重算到容量公式"
date: 2026-09-02 09:00 +0800
tags: [vLLM, LLM 推理, KV Cache, GQA, Tensor Parallel]
toc: true
math: true
permalink: /vllm/kv-cache/
---

## 本篇要回答什么

自回归模型已经生成过的 token，为什么还需要保存 K 和 V？缓存究竟省掉了什么，为什么长上下文 decode 仍然可能很慢？单卡、多卡、MHA、GQA 和 MQA 下，KV Cache 容量又应怎样计算？

核心结论是：**KV Cache 用显存和历史 K/V 读取，换掉旧 token 在后续 decode step 中重复经过所有 Transformer 层的计算。**它消除历史位置的重复投影与 MLP，却没有消除新 query 对可见历史的注意力读取。

## 0. K 和 V 到底怎样产生

先只看一个 decoder attention 层。上一层交给它的隐藏状态记作

$$
X\in\mathbb{R}^{B\times T\times H},
$$

其中 $B$ 是 batch size，$T$ 是本轮实际处理的 token 数，$H$ 是 hidden size。经过归一化后，对同一个 $X$ 做三组线性投影：

$$
Q=XW_Q+b_Q,\qquad K=XW_K+b_K,\qquad V=XW_V+b_V.
$$

若有 $N_q$ 个 query heads、$N_{kv}$ 个 KV heads、每个 head 的维度为 $D$，权重与输出 shape 是：

| 对象 | shape | 最后一维的含义 |
|---|---|---|
| $W_Q$ | $[H,N_qD]$ | 所有 query heads 拼接 |
| $W_K$ | $[H,N_{kv}D]$ | 所有 key heads 拼接 |
| $W_V$ | $[H,N_{kv}D]$ | 所有 value heads 拼接 |
| $Q$ | $[B,T,N_q,D]$ | reshape 后的 queries |
| $K,V$ | $[B,T,N_{kv},D]$ | reshape 后的 keys/values |

工程实现通常把三次 GEMM 融成一次：

$$
[Q\mid K\mid V]=XW_{QKV},
$$

其中 $W_{QKV}$ 在输出维拼接三组权重。**fused QKV projection 只减少 kernel launch 和输入读取，不表示 Q、K、V 的语义相同，也不表示它们共享权重。**投影结果仍会按各自长度切开并 reshape。

下面是一个可以独立运行的 shape 演示：

```python
import torch

B, T, H = 2, 3, 16
N_Q, N_KV, D = 4, 2, 4       # GQA：两个 query heads 共享一个 KV head
x = torch.randn(B, T, H)
w_qkv = torch.randn(H, (N_Q + 2 * N_KV) * D)

qkv = x @ w_qkv
q_end = N_Q * D
k_end = q_end + N_KV * D
q = qkv[..., :q_end].view(B, T, N_Q, D)
k = qkv[..., q_end:k_end].view(B, T, N_KV, D)
v = qkv[..., k_end:].view(B, T, N_KV, D)

assert q.shape == (2, 3, 4, 4)
assert k.shape == v.shape == (2, 3, 2, 4)
print(q.shape, k.shape, v.shape)
```

输入是隐藏状态和一张教学用 fused 权重；输出是拆分、reshape 后的 Q/K/V。代码不含 bias、Q/K norm、量化权重、TP 切分和真实 vLLM 的 packed parameter loader，所以只能证明 shape 与切片关系，不能证明生产 kernel 的布局或性能。

### RoPE 为什么作用于 Q、K，而通常不作用于 V

注意力分数来自 $QK^T$，位置信息必须进入“某个 query 与某个 key 的匹配关系”。旋转位置编码（Rotary Position Embedding，RoPE）对位置 $i,j$ 的 Q/K 施加旋转：

$$
\tilde q_i=R_iq_i,\qquad \tilde k_j=R_jk_j,
$$

从而

$$
\tilde q_i^T\tilde k_j=q_i^TR_i^TR_jk_j=q_i^TR_{j-i}k_j.
$$

分数自然依赖相对位置 $j-i$。V 是分数确定以后被加权汇总的“内容”；若也旋转 V，输出内容会额外依赖其绝对位置，而且不能仅凭同一组 attention weights 恢复原内容。因此常见 RoPE decoder 对 Q、K 旋转而不旋转 V。这里说的是常见结构，不是所有位置编码方案的定律。

## 1. 模型处理一个 token，预测的是下一个 token

设 prompt 有 $P=4$ 个 token，需要生成 $G=5$ 个 token $y_1,\ldots,y_5$。prefill 处理完整 prompt 后直接产生 $y_1$；之后处理 $y_1$ 才产生 $y_2$。

| 模型处理的输入 | 新得到的输出 |
|---|---|
| 4 个 prompt token | $y_1$ |
| $y_1$ | $y_2$ |
| $y_2$ | $y_3$ |
| $y_3$ | $y_4$ |
| $y_4$ | $y_5$ |

请求在 $y_5$ 返回后结束，不必再把 $y_5$ 输入模型，否则得到的是并不需要的 $y_6$。因此缓存路径实际处理的位置数是

$$
P+(G-1)=4+4=8.
$$

没有缓存时，为生成每个输出都要重新送入完整已有前缀，总位置数是

$$
\sum_{g=0}^{G-1}(P+g)
=GP+\frac{G(G-1)}2
=4+5+6+7+8=30.
$$

若用稠密 attention score 数量做教学估算，无缓存为

$$
4^2+5^2+6^2+7^2+8^2=190,
$$

缓存路径则是一次 $4\times4$ prefill，加上四个新 query 分别读取长度 5、6、7、8 的历史：

$$
4^2+(5+6+7+8)=42.
$$

这些数字说明依赖规模，不等同于 profiler 的真实 FLOPs、HBM byte 或延迟。高性能 causal kernel 不一定物化完整方阵。

## 2. 每层究竟缓存什么

对普通 MHA/GQA decoder，一层的逻辑 KV shape 可以写成：

```text
K: [tokens, num_kv_heads, head_dim]
V: [tokens, num_kv_heads, head_dim]
```

设：

- $L$：需要缓存的层数；
- $T$：缓存 token 数；
- $N_{kv}$：KV head 数；
- $D$：每个 head 的维度；
- $e$：每个元素的 byte 数。

普通 KV payload 为

$$
C_{KV}=2LTN_{kv}De.
$$

最前面的 2 代表 Key 和 Value。以 28 层、8 个 KV heads、head dimension 128、BF16 为例，每 token 为

$$
2\times28\times8\times128\times2
=114688\ \text{bytes}
=112\ \text{KiB}.
$$

4096 tokens 的理论 payload 是 448 MiB。这仍未包含 block 对齐、scale、allocator、graph buffer 和 workspace。

## 3. MHA、GQA、MQA 改变的是 $N_{kv}$

Query heads 与 KV heads 不必相等：

| 结构 | KV heads | KV payload 特征 |
|---|---:|---|
| MHA | 通常 $N_{kv}=N_h$ | 每个 query head 有独立 K/V |
| GQA | $1<N_{kv}<N_h$ | 多个 query heads 共享一组 K/V |
| MQA | $N_{kv}=1$ | 所有 query heads 共享一组 K/V |

若 query heads 为 16，而三种配置的 $N_{kv}$ 分别为 16、8、1，在其余条件相同时 payload 比例是 $16:8:1$。这只是容量比例，不是延迟或吞吐比例；query 行数、投影、通信和 kernel 路径没有按同一比例消失。

也不能普遍用 hidden size $H$ 代替 $N_{kv}D$。某个模型恰好满足二者相等，只是配置巧合；GQA 模型常有 $N_{kv}D<H$。

以 $B=2,T=128,N_q=16,D=64$ 为例：

| 结构 | $N_{kv}$ | Q shape | K/V shape | 每个 KV head 服务几个 Q heads |
|---|---:|---|---|---:|
| MHA | 16 | `[2,128,16,64]` | `[2,128,16,64]` | 1 |
| GQA | 4 | `[2,128,16,64]` | `[2,128,4,64]` | 4 |
| MQA | 1 | `[2,128,16,64]` | `[2,128,1,64]` | 16 |

计算时可把 KV heads 逻辑广播到对应的 query groups，但缓存里没有必要真的复制为 $N_q$ 份。缓存容量因此跟 $N_{kv}$ 相关，而 attention 输出仍有 $N_q$ 个 heads。

## 3.1 Prefill、decode 与缓存写入的完整时间线

对长度为 $P$ 的 prompt，prefill 一次产生 $P$ 个位置的 Q/K/V。causal mask 使位置 $i$ 只能读到 $0..i$；所有层的 K/V 被写入缓存，最后一个 prompt 位置的 logits 用来采样 $y_1$。

下一轮 decode 只把 $y_1$ 的 hidden state 送过各层：

```text
本层输入 x_new [B,1,H]
  → Q_new [B,1,N_q,D]
  → K_new,V_new [B,1,N_kv,D]
  → 对 Q_new/K_new 应用当前位置的 RoPE
  → 把 K_new,V_new 写入该请求的新 slot
  → Q_new 读取 K_cache,V_cache 的全部可见历史
  → attention output [B,1,N_q,D]
  → output projection + residual + MLP
```

无缓存与有缓存的教学伪代码分别是：

```python
# 教学伪代码：无缓存。每一轮重新处理完整 prefix。
tokens = prompt[:]
for _ in range(max_new_tokens):
    logits = model.forward(tokens)       # 长度 P, P+1, P+2, ...
    new_token = sample(logits[-1])
    tokens.append(new_token)

# 教学伪代码：有缓存。prefill 后每轮只投影一个新位置。
logits, cache = model.prefill(prompt)    # 写入 prompt 的 K/V
tokens = prompt[:]
for step in range(max_new_tokens):
    new_token = sample(logits[-1])
    tokens.append(new_token)
    if step + 1 == max_new_tokens:
        break                            # 最后输出不用再送回模型
    logits, cache = model.decode(new_token, cache)
```

输入都是 prompt 和最大输出长度；输出是生成 token。第二段的长期状态是逐层 cache，shape 从每层 `[P,N_kv,D]` 增长到 `[P+step,N_kv,D]`。它对应前文的 $P+G-1$ 位置计数。伪代码省略 batch、EOS、采样参数、分页 slots、推测 token、抢占和分布式通信，不能用于测量真实加速比。

## 4. 多卡时必须计算 rank-local KV

分布式部署中，真正占用某张卡的是本 rank 保存的层和 KV heads：

$$
C_{rank}=2L_{local}T_{local}N_{kv,local}De.
$$

Tensor Parallelism（TP）通常切分同一层的 heads。若 $N_{kv}\ge p$ 且可整除，TP size 为 $p$ 时每 rank 保存 $N_{kv}/p$ 个 KV heads。但当 TP ranks 多于 KV heads 时，一个 head 不能被切成零点几份，具体实现可能复制 KV heads，使每 rank 至少有一组。

以 8 个 KV heads 为例，vLLM v0.20.0 的 Qwen3 路径可概括为：

| TP | 每 rank KV heads | BF16 KiB/token（28 层、$D=128$） |
|---:|---:|---:|
| 1 | 8 | 112 |
| 2 | 4 | 56 |
| 4 | 2 | 28 |
| 8 | 1 | 14 |
| 16 | 1（复制） | 14 |

因此“所有显存都除以 TP size”并不可靠。

Pipeline Parallelism（PP）切的是层，所以公式中的 $L$ 应换成本 stage 的 $L_{local}$。Data Parallelism（DP）通常复制服务实例并把不同请求分给 replicas，不会自动把单请求 KV 分散到两个副本，也不会把不同副本的空闲 block 合成一个共享池。

### global rank、local rank 与 TP rank

rank 是参加分布式计算的进程编号，常见部署是一 worker process 管一张 GPU，但概念上二者仍不同：

- global rank：整个作业中的编号，范围 `0..world_size-1`；
- local rank：当前机器上的编号，常用于选择本机 GPU；
- TP rank：当前 tensor-parallel group 内的编号；
- PP rank：当前 pipeline 中的 stage 编号；
- DP rank：当前 replica 编号。

例如 DP=2、每个副本 TP=4 时，global ranks 可以是 `[0,1,2,3]` 和 `[4,5,6,7]` 两个 TP group。global rank 5 的 TP rank 是 1，而不是 5。若再引入 PP，不能再用 GPU 编号猜它保存哪些层或 heads。

PP 传的是 stage 间 hidden states，而不是把前一 stage 的 KV 搬到后一 stage：

```text
token → PP stage 0（本地层 + 本地 KV）
      → hidden states [scheduled_tokens,H]
      → PP stage 1（本地层 + 本地 KV）→ logits
```

vLLM 推理只做前向，不做训练反向传播。训练时才会把 $\partial loss/\partial h$ 沿 PP 反向传回，并在 DP group 同步参数梯度；这不属于 KV Cache serving 主线。

下面的容量计算器把全局配置转换成每 rank 的 payload：

```python
from math import ceil

def local_kv_heads(total_kv_heads: int, tp: int) -> int:
    if total_kv_heads >= tp:
        assert total_kv_heads % tp == 0
    else:
        assert tp % total_kv_heads == 0
    return max(1, total_kv_heads // tp)

def kv_bytes(*, layers, tokens, kv_heads, head_dim,
             bytes_per_elem=2, tp=1, pp=1, block_size=None):
    assert layers % pp == 0
    local_layers = layers // pp
    local_heads = local_kv_heads(kv_heads, tp)
    allocated_tokens = tokens
    if block_size is not None:
        allocated_tokens = ceil(tokens / block_size) * block_size
    per_rank = (2 * local_layers * allocated_tokens
                * local_heads * head_dim * bytes_per_elem)
    return {
        "local_layers": local_layers,
        "local_kv_heads": local_heads,
        "allocated_tokens": allocated_tokens,
        "bytes_per_rank": per_rank,
    }

qwen = kv_bytes(layers=28, tokens=4096, kv_heads=8,
                head_dim=128, tp=4, pp=2, block_size=16)
assert qwen["local_layers"] == 14
assert qwen["local_kv_heads"] == 2
assert qwen["bytes_per_rank"] == 56 * 1024**2  # 56 MiB
print(qwen)
```

输入是模型结构、token 数、dtype byte、TP/PP 和可选 block size；输出是本 rank 的层数、heads、取整 slots 和 payload。状态不发生变化。公式就是 $2L_{local}T_{allocated}N_{kv,local}De$。真实 vLLM 还会按 cache group/spec、非均匀 PP 层分配、scale、对齐和可用显存建立 pool，因此这只是常规 attention 的容量核算器。

## 5. 有效 KV、block-rounded KV 与 KV pool

假设 block size 为 16，两个请求长度为 17 和 32：

| 请求 | 有效 token | 分配 block | 分配 slots | 末块未用 |
|---|---:|---:|---:|---:|
| A | 17 | 2 | 32 | 15 |
| B | 32 | 2 | 32 | 0 |
| 合计 | 49 | 4 | 64 | 15 |

这会产生三个不同的容量口径：

```text
有效 KV：           49 × 每 token byte
block-rounded KV：  64 × 每 token byte
KV pool：           引擎启动时预留的全部物理 blocks
```

请求释放只会让 blocks 回到 vLLM 的空闲队列，整个 pool 未必归还 CUDA。因此一次请求前后的 `memory_allocated()` 差值不能精确表示该请求的有效 KV。

把前面的 Qwen3 数值代入：每 token 112 KiB，49 个有效 tokens 是

$$49\times112\ \mathrm{KiB}=5488\ \mathrm{KiB}\approx5.36\ \mathrm{MiB},$$

而 64 个已分配 slots 是 7 MiB，末块 15 个空 slots 占约 1.64 MiB。KV pool 可能是数 GiB；“池已占显存”“某请求已获 block”“slot 已写入有效 K/V”是三个不同状态。

## 6. KV Cache 省了什么，没省什么

第 $g$ 个 decode step 的逻辑 shape 仍近似为：

```text
Q_new: [batch, num_query_heads, 1, head_dim]
K/V:   [batch, num_kv_heads, history, head_dim]
```

新 query 仍要读取可见历史 K/V。于是 KV Cache 的收益与成本应分开记账：

- 省掉：旧 token 的 RMSNorm、QKV/输出投影、MLP 与旧 query attention 的重复执行；
- 保留：当前 query 对历史 K/V 的读取与 attention；
- 新增：长期 KV 容量、prefill/decode 写入、block 管理与正确性约束；
- 可能出现的新瓶颈：HBM、kernel launch、CPU 调度或跨卡通信。

$G=1$ 时只有一次 prefill，没有后续 step 可复用，缓存路径在计算位置数上没有优势。长上下文与高并发会放大重算收益，也同时放大 KV 容量和带宽压力。

再看一个长序列例子：$P=2048,G=128$，无缓存的投影位置总数为

$$128\times2048+\frac{128\times127}{2}=270272,$$

有缓存则为 $2048+127=2175$，相差约 124.26 倍。这不是 124.26 倍 wall-clock 加速：decode 的窄 GEMM、每层权重读取、历史 KV 的 HBM 流量、调度、IPC、collective 和 kernel launch 都还存在。

### 特殊解码的 cache 生命周期

- beam search：分叉前的 blocks 可共享，分叉后每条 beam 独立增长；引用计数必须避免过早释放共享前缀；
- speculative decoding：候选 token 可暂时产生 KV，最终只提交被接受的前缀，并丢弃/回滚其余候选；
- sliding window：逻辑上只读最近窗口，但物理 block 能否立刻回收取决于 cache spec 与 backend；
- prompt logprobs：可能要求保留/计算不同位置的 logits，不能套用只取最后位置的最小时间线；
- sequence continuation：若服务要为下一次续写保留最后输出的已计算状态，停止边界可能从 $P+G-1$ 变为 $P+G$。

### 四种容易混淆的 KV 技术

| 技术 | 复用范围 | 数据位置变化 | 主要目标 |
|---|---|---|---|
| 普通 KV Cache | 同一请求的后续 decode | 通常不变 | 避免历史重算 |
| APC | 兼容请求之间的完整前缀块 | 通常不变 | 跳过命中前缀 prefill |
| KV offload | 同一 cache 生命周期 | GPU 与 CPU/慢层级之间 | 换取 GPU 容量 |
| KV transfer / P-D 分离 | prefill 节点到 decode 节点 | 跨进程/节点传输 | 避免 decode 节点重算 prefill |

关闭 APC 不会关闭普通 KV Cache；APC 命中也不会缓存最终答案。

## 7. 公式的适用边界

公式 $2LTN_{kv}De$ 针对普通、每层显式保存 K/V 的 attention。以下情况应先查看模型配置和 vLLM 生成的 cache spec：

- MLA 缓存压缩 latent 与位置相关分量；
- sliding-window 层只需逻辑可见窗口，但物理回收依赖 backend；
- hybrid model 可能混合 full、sliding 与 state-space cache；
- encoder-decoder 还有 cross-attention KV；
- KV 量化带有额外 scale；
- offload、transfer 和 P/D disaggregation 改变存储位置与传输成本。

## 常见误区

- “KV Cache 把生成从 $O(n^2)$ 变成 $O(n)$”：必须说明对哪个阶段、变量和成本项而言。
- “GQA 的 KV 小四倍，所以服务快四倍”：容量比例不能直接变成速度比例。
- “TP=8，所以单卡 KV 一定除以 8”：KV heads 不足时可能复制。
- “关闭 APC 就关闭了 KV Cache”：APC 是跨请求前缀复用，普通自回归 KV Cache 仍然存在。

## 源码阅读入口（v0.20.0）

- `vllm/model_executor/models/qwen3.py`：全局与本地 query/KV heads；
- `vllm/v1/kv_cache_interface.py`：不同 attention 类型的 cache spec；
- `vllm/config/cache.py`：cache dtype 与容量配置；
- `vllm/v1/core/kv_cache_manager.py`：请求如何取得和释放物理 blocks。

## 本篇总结

KV Cache 的正确心智模型不是“保存所有中间结果”，而是每层保存可供未来 query 使用的历史 K/V。它显著减少历史位置的重复层计算，但把系统推向新的容量、带宽和生命周期问题；PagedAttention、Prefix Caching 与 Scheduler 正是对这些问题的进一步回答。

---

[系列首页]({{ '/vllm/' | relative_url }}) · [下一篇：vLLM V1 架构]({{ '/vllm/architecture/' | relative_url }})

## 资料

- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
