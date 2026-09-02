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
