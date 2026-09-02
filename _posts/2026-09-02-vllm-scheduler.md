---
layout: post
title: "vLLM Scheduler：Continuous Batching 与 Chunked Prefill"
date: 2026-09-02 09:40 +0800
tags: [vLLM, LLM 推理, Scheduler, Continuous Batching, Chunked Prefill]
toc: true
math: true
permalink: /vllm/scheduler/
---

## 本篇要回答什么

请求到达时间、prompt 长度和输出长度都不同，GPU 却希望持续获得足够工作。vLLM Scheduler 的任务是在每次 iteration 边界重新决定：哪些请求进入本轮、各推进多少 tokens、KV blocks 是否够，以及容量不足时谁被抢占。

## 1. 三种 batch

- client batch：一次 API 调用提交的 prompts；
- static/request batch：一段时间内成员固定的 batch；
- iteration batch：某次模型执行真正打包的请求和 tokens。

Continuous Batching（连续批处理）缩短的是成员决策周期：模型每执行一轮，已完成请求可以退出，新请求可以在下一轮补入。

```text
static batch:
[A B] ─────────────────── 等 A、B 都结束后才能换成员

continuous batching:
step 0 [A B]
step 1 [A B]
step 2 [A C]   B 已完成，C 补入
```

C 落在 step 2 还是 3，取决于它在本轮 schedule 之前还是之后到达；画时间线时必须声明事件边界。

iteration batch 也不一定是简单的 `[batch, sequence]` 矩形：decode 请求可能贡献 1 token，partial prefill 贡献数百 tokens，speculative path 贡献多个候选位置。

## 2. 请求状态不等于 prefill/decode

`WAITING` 与 `RUNNING` 表示调度生命周期，不是模型阶段的别名。长 prompt 被接纳并计算第一个 chunk 后已经是 RUNNING，但仍在做 prefill；请求也可能等待 grammar、remote KV、streaming 或处于 PREEMPTED/FINISHED 状态。

vLLM V1 的统一视角是比较逻辑 token 数与已计算进度。在省略 async placeholder 等分支的普通情形下：

$$
deficit=
\text{num\_tokens\_with\_spec}
-\text{num\_computed\_tokens}.
$$

这一个差额能表达多种工作：

| 场景 | deficit 的含义 |
|---|---|
| 10-token prompt，computed=0 | 仍需 prefill 10 |
| 已完成前 4 个 prompt tokens | 仍需 prefill 6 |
| APC 命中 8 个 tokens | 只追赶未命中后缀 |
| 刚采样 1 个 token | 通常差 1，处理它以产生下一个 |
| 添加 draft tokens | 一轮待验证位置增加多个 |

Scheduler 内部因此不必把所有路径硬拆成独立的 prefill loop 和 decode loop。

## 3. token budget 不是输出长度

token budget 是本 iteration 可以处理的 token 工作量上限；`max_tokens` 是某请求最多生成多少输出。二者回答不同问题。

即使 budget 有余，请求仍可能因以下约束无法执行：

- KV blocks 不足；
- maximum model length；
- running request 数量上限；
- encoder budget/cache；
- LoRA 限制；
- alignment、connector 或 structured-output 状态。

同样，sequence slot 有空也不代表还有足够 token 或 KV 容量。

## 4. 三个请求的逐轮时间线

固定一个教学策略：每轮 token budget=6，单请求 prefill chunk 上限=4；先安排 RUNNING，再按 FCFS 接纳 WAITING。

| 请求 | 到达 | prompt $P$ | output $G$ |
|---|---:|---:|---:|
| A | step 0 | 4 | 2 |
| B | step 0 | 8 | 3 |
| C | step 1 | 3 | 2 |

```text
step 0: {A:4, B:2}
  A prefill 完成并采样 A1；B computed=2/8

step 1: {A:1, B:4, C:1}
  A 处理 A1、采样 A2 后完成；B 到 6/8；C 到 1/3

step 2: {B:2, C:2}
  B、C 都完成 prompt，各采样第一个输出
  剩余 budget=2 不能“回到本轮输入”处理刚刚才采样出的 token

step 3: {B:1, C:1}
  B 采样第二个输出；C 采样第二个输出并完成

step 4: {B:1}
  B 采样第三个输出并完成
```

每个请求真正送入模型的位置数为 $P+G-1$：

$$
A=5,\quad B=10,\quad C=4,\quad total=19.
$$

各轮 scheduled totals 也是 $6+6+4+2+1=19$。完成时每个请求的逻辑 token 数通常比 computed 多 1，因为最终采样 token 无需再次送入模型。

## 5. `Scheduler.schedule()` 的主干

生产代码分支很多，第一次阅读可以先固定普通 decoder-only、单 KV group、无 connector/spec/multimodal 的子集：

```text
1. 初始化本轮 collections 与 budgets
2. 遍历 RUNNING，请求先追赶自己的 deficit
3. KV 不足时选择 victim、preempt，再重试分配
4. 若没有被抢占阻断，使用剩余资源准入 WAITING
5. 构造 SchedulerOutput
6. 推进 num_computed_tokens 等进度，等待执行结果返回
```

RUNNING 优先通常保护已有 stream 的 ITL，但 RUNNING 中也可能有 partial prefill，不能简单把这段命名为 decode loop。

`allocate_slots` 成功后才应记录 blocks、scheduled token 数并扣减 budget。容量不足时需要撤销受害请求已安排的本轮记录，再改变其状态和资源。FCFS/priority 下的 victim 选择必须按固定版本代码判断，不能概括成永远抢占“最新”或“最长”请求。

WAITING 准入还要区分新请求和被抢占后需要 recompute 的请求，并结合 APC/remote KV 的 computed progress。Chunking 关闭时，过长的未计算输入可能不能用本轮剩余小 budget 部分接纳；开启后才可分块推进。

## 6. Chunked Prefill 改变粒度，不减少总工作

本轮 budget=2048，100 个 running decode requests 各需要 1 token，剩余 1948。一个长 prompt 尚有 6000 tokens 未计算：

```text
本轮：100 decode + 1948 prompt tokens
下轮：继续安排剩余 prompt tokens
```

长 prompt 最终仍计算 6000 tokens，也仍要保存相应 KV。Chunked Prefill 只改变它分几轮执行。

| 选择 | 可能的收益 | 可能的代价 |
|---|---|---|
| 小 chunk | decode 更频繁获得机会，tail ITL 可能改善 | 更多 iterations、launch 和调度；长请求 TTFT/input throughput 可能变差 |
| 大 chunk | prefill GEMM 更饱满，长请求更快到首 token | 单轮更长，已有 streams 等待，P99 ITL 和 activation 峰值可能增大 |

拐点依模型、GPU、并发、backend 和 shape 而变，不能从机制直接推出所有用户都更快。

Continuous Batching 决定请求成员能否跨轮变化；Chunked Prefill 决定一个 prompt 能否跨轮处理。二者可组合但不是同义词。

## 7. 调度不是越满越好

更大的 token budget 和更多 running requests 可能提高吞吐与权重读取摊销，也可能：

- 增加 queue time 或单轮 device time；
- 消耗更多 KV blocks；
- 让 TTFT、ITL 的尾部更差；
- 扩大动态 shape，降低某些 graph bucket 的命中；
- 更频繁触发 preemption/recompute。

评估时至少同时记录 input/output token throughput、TTFT、ITL/TPOT、E2E 分位数、queue time、preemption 和 KV 使用。

## 常见误区

- “Continuous Batching 就是把 batch size 调大”：它强调 iteration 边界动态换成员。
- “每个请求每轮只算一个 token”：prefill chunk 和 speculative tokens 可以贡献多个位置。
- “WAITING 是 prefill，RUNNING 是 decode”：状态与计算阶段不能一一翻译。
- “Chunked Prefill 减少 prefill FLOPs”：它改变粒度和竞争方式，不减少总 prompt 计算。

## 源码阅读入口（v0.20.0）

- `vllm/v1/core/sched/scheduler.py`：`Scheduler.schedule()` 主干；
- `vllm/config/scheduler.py`：token budget、chunked prefill 与相关限制；
- `vllm/v1/request.py`：请求状态与 token progress；
- `vllm/v1/core/kv_cache_manager.py`：调度时的 block 查找和分配。

## 本篇总结

Scheduler 的统一语言不是“这轮是 prefill 还是 decode”，而是请求当前拥有多少逻辑 tokens、已有多少被模型计算，以及本轮 budget 和 KV 容量能填补多少 deficit。Continuous Batching 动态重组成员，Chunked Prefill 拆分长输入；两者共同在吞吐、TTFT 与 ITL 之间做可测量的权衡。

---

[上一篇：Prefix Caching]({{ '/vllm/prefix-caching/' | relative_url }}) · [系列首页]({{ '/vllm/' | relative_url }}) · [下一篇：量化与推测解码]({{ '/vllm/quantization-speculative-decoding/' | relative_url }})

## 资料

- [vLLM v0.20.0 Scheduler 源码](https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/v1/core/sched/scheduler.py)
- [vLLM 官方优化与调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)
