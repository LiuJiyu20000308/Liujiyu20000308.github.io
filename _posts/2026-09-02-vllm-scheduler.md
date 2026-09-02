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

静态 batch 的浪费有两层。prompt 长度 `[4,4,20]` 若做矩形输入，会有 $3\times20-28=32$ 个 padding positions；输出长度 `[2,8,2]` 若成员固定，A/C 完成后对应 rows 还会等待 B。高效 kernel 可用 mask/packing 避免一部分 padding 计算，但“短请求不能及时退出并让新请求加入”仍是调度问题。

请求到达边界必须精确到 schedule：

```text
iteration k execute ───────┐
                           ├─ result/update ─ schedule(k+1) ─ execute
C 在 schedule 前到达 ─────┘                 ↑ 可进入 k+1
D 在 schedule 后到达 ──────────────────────┘ 只能等 k+2
```

Continuous Batching 不承诺零等待，只把重新选择成员的机会缩短到 iteration 边界。

一次 iteration 内已经打包好的 GPU batch 通常要作为一个执行单元完成，Scheduler 才能可靠消费结果并决定下一轮成员。它不是在 kernel 执行一半时把某一 row 抽走；动态性发生在 iteration 之间。异步/batch-queue 路径可以重叠 CPU 准备和相邻批次，但仍要维护明确的数据依赖。

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

完整 deficit 还要考虑“逻辑上已加入但尚未验证”的 speculative tokens，以及异步路径的 placeholders。可以把三个量分开：

- `all_token_ids`：已确认的 prompt + output token 序列；
- `spec_token_ids`：proposer 给出的待验证候选；
- `num_computed_tokens`：目标模型/KV 已实际推进的位置。

因此 `num_tokens_with_spec - num_computed_tokens` 是“当前可安排工作”，而不是剩余输出长度。`PREEMPTED` 请求可能已有逻辑 tokens 却因 KV 被回收而需要 recompute；`WAITING` 请求也可能因为 APC/remote KV 已有非零 computed progress。

这里还要修正变量名带来的误解：v0.20.0 的 Scheduler 在构造 `SchedulerOutput` 后，会用 `_update_after_schedule` 把 `num_scheduled_tokens` 加入 `num_computed_tokens`，此时 GPU 可能尚未物理完成。对 Scheduler 而言它更接近“已完成或已提交、下一轮不能重复安排的计算前沿”。同步路径在下一次消费该进度前通常已完成执行；spec rejection、KV load 失败或异步结果会在 output 处理时校正。

```text
旧 computed=4
  → 构造 plan：从 position 4 开始处理 4 个
  → Scheduler 账面 computed=8（4..7 已被本轮认领）
  → executor/GPU 真正执行
  → result 校正/采样
```

必须先用旧进度构造 plan；若先改为 8，worker 可能错误地从 position 8 开始。

### async output placeholder

`num_output_placeholders` 表示“已启动的异步采样将产生、但真实 token id 尚未并入 Request 的输出位置”，不是假 token，也不是结束标志。10-token prompt 按 4/4/2 做 prefill，只有最后 2 个追平 prompt、将产生首个输出时才增加 placeholder；前两个 chunk 不产生输出。

结果写回 token 42 时：

```text
写回前：num_tokens_with_spec=10, placeholder=1, effective_frontier=11
写回后：num_tokens_with_spec=11, placeholder=0, effective_frontier=11
```

真实 token 数增加 1、placeholder 减少 1，有效逻辑前沿不凭空增长。结束由 EOS、stop、max tokens、abort/error 等真实条件决定，不由 placeholder 值决定。异步流水可能在 EOS 结果揭晓前提前提交下一轮，造成少量无用 GPU 工作；正确实现必须丢弃晚到结果、不能把 EOS 后 token 交给用户，并最终释放资源。

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

三个资源的单位不同：

| 资源 | 单位 | 用尽后的表现 |
|---|---|---|
| token budget | 本 iteration 的 scheduled positions | 剩余请求等下一轮 |
| sequence slots | 同时驻留/打包的请求数 | 有 budget 也不能再接请求 |
| KV blocks | pool 中的物理 blocks | 分配失败、等待或抢占 |

`max_tokens=128` 是单请求采样停止条件；`max_num_batched_tokens=2048` 一类预算约束本轮执行规模。二者即使数值相同也没有同一语义。

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

新采样 token 不能在同一 iteration“用剩余 budget 立即喂回”，因为本轮 model forward 的 inputs、positions、attention metadata 和 CUDA 工作在采样前已经确定。把输出再作为输入需要构造下一张执行图/下一批 tensor，这正是下一个 schedule 边界。推测解码是在本轮开始前已有多个 draft positions，不是 forward 结束后递归复用同一批次。

### 可执行的简化 Scheduler simulator

```python
from dataclasses import dataclass

@dataclass
class Req:
    name: str
    arrival: int
    prompt: int
    output: int
    computed: int = 0
    sampled: int = 0

    @property
    def logical(self):
        # prompt 加已采样 token；最后采样 token尚未 computed
        return self.prompt + self.sampled

    @property
    def done(self):
        return self.sampled == self.output

reqs = [Req("A",0,4,2), Req("B",0,8,3), Req("C",1,3,2)]
budget, chunk = 6, 4
history = []

for step in range(5):
    left, plan = budget, {}
    for r in reqs:                    # 教学版 FCFS
        if r.arrival > step or r.done or left == 0:
            continue
        deficit = r.logical - r.computed
        if deficit == 0:
            continue
        n = min(deficit, chunk if r.computed < r.prompt else 1, left)
        plan[r.name] = n
        left -= n
    # execute：只推进本轮开始前已经存在的 positions
    for r in reqs:
        if r.name not in plan:
            continue
        r.computed += plan[r.name]
        # prompt 或上一个 sampled token被追平后，本轮产生一个新 token
        if r.computed == r.logical and r.sampled < r.output:
            r.sampled += 1
    history.append(plan)

assert history == [
    {"A": 4, "B": 2},
    {"A": 1, "B": 4, "C": 1},
    {"B": 2, "C": 2},
    {"B": 1, "C": 1},
    {"B": 1},
]
assert sum(sum(p.values()) for p in history) == 19
assert all(r.done for r in reqs)
```

输入是到达轮次、prompt/output 长度、budget 与 chunk；输出是每轮 request→scheduled tokens。长期状态为 `computed/sampled`。它验证 19-token 守恒与“采样结果下一轮再输入”。它没有 KV blocks、真实 RUNNING/WAITING 队列、APC、spec decode、并行、优先级和失败回滚，不能代表生产公平性或性能。

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

### KV 分配失败与 rollback

一个安全的教学主干是“先试资源，成功后提交计划；要抢占时撤销本轮临时记录”：

```text
# 教学伪代码
for req in running_in_policy_order:
    n = min(req.deficit, token_budget)
    while n > 0:
        blocks = kv_manager.allocate_slots(req, n)
        if blocks is not None:
            plan.commit(req, n, blocks)
            token_budget -= n
            break

        victim = choose_victim(running, policy)
        if victim is None:
            n = 0
            break
        plan.rollback_if_scheduled(victim)
        kv_manager.free(victim)
        victim.num_computed_tokens = recompute_boundary(victim)
        victim.status = PREEMPTED
        move_to_waiting(victim)

if not preemption_blocked_admission:
    admit_waiting_with_remaining_resources()
return SchedulerOutput.from_plan(plan)
```

输入是权威队列、token budget 与 KV pool；输出是本轮不可变计划。长期状态可能改变队列、请求状态、computed progress、block ownership。真实 victim policy、是否 swap/recompute、connector 状态和调度顺序必须按 v0.20.0 固定代码核对，不能把伪代码中的名字当成真实函数。

为什么要 rollback：假设 victim 先被记入 plan 2 tokens，随后为了另一个请求释放它的 blocks；若不同时撤销 plan 和返还 budget，worker 会收到一个引用已释放 blocks 的计划，或者守恒式重复扣预算。

### APC、remote KV 与 recompute 怎样进入统一进度

- APC：等待请求 lookup 到 $h$ 个连续命中 tokens，可把 computed 起点推进到安全命中边界；
- remote KV：connector 可能让请求等待传输/加载完成，完成前不能把远端数据当作本地可读；
- recompute：抢占后物理 KV 已释放，逻辑 tokens 仍在，但 computed 可能退回可重建边界；
- speculative：本轮目标模型实际接受几个 token，要在 runner output 返回后才能提交。

共同原则是：在 iteration 边界，`num_computed_tokens` 必须对应可复用或已按计划提交、且有失败校正机制覆盖的计算前沿；不能把“过去某时算过但 KV 已释放”或“尚未进入有效计划的愿望”算进去。

APC lookup 的频率也要区分：NEW/WAITING 请求每次尝试准入时可查询最长缓存前缀；若资源不足仍留在 WAITING，下轮重试时 cache 状态可能变化，所以可再查；PREEMPTED 请求重新准入也会再查。已经 RUNNING 的请求通常沿用自己绑定的 blocks，不会每轮重新搜索 APC，但仍会每轮分配/提交/释放 KV metadata。

### `new_reqs`、FCFS 与 full-ISL reserve

`NewRequestData`/`new_reqs` 记录本轮首次同步给 worker-side runner 的请求；cached/running request 只需传增量。worker 需要这一区分来创建本地 request row、初始化 token/position/block-table 状态，避免每轮重发完整对象。它描述“本轮执行计划里的新成员”，不是预先保存下一轮请求。

FCFS（First-Come, First-Served，先到先服务）按到达/排队顺序考虑请求；priority policy 则还比较优先级与到达时间。KV 不足时“让当前请求原地等待”并不能释放任何 block，若希望它继续而 pool 已满，只能选择某个活跃 victim 释放，或本轮放弃当前请求。一个请求释放后仍不够，Scheduler 可以继续选择更多 victims，直到分配成功或当前请求本身也无法保留。

Chunked Prefill 只看首 chunk 容量时可能过度准入很多长 prompt：每个先拿 128 tokens 的 blocks，后续增长又争夺 pool，形成抢占—重算 thrashing。`scheduler_reserve_full_isl`（ISL，Input Sequence Length）尝试在准入时按完整输入长度预留/检查，而不是说“每次 chunk 都占同样 128”。概念上：

```text
tokens_to_schedule = min(本轮剩余输入, chunk limit, token budget)
tokens_to_reserve  = full ISL 对应容量（开启 reserve 时）
```

`tokens_to_reserve` 由请求长度、computed/cached 进度和配置逻辑计算，不是用户随意再写一个常数。字段名和具体公式应从 v0.20.0 `SchedulerConfig` 与 `schedule()` 分支核对。reserve 更保守，可能降低并发/增加 queue time，也不能覆盖未来无限 output 增长。

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

P/D disaggregation（prefill/decode 分离）又是第三件事：它把两个阶段放到不同 worker/节点，并传输 KV；Continuous Batching 与 Chunked Prefill 仍可在各自节点内部使用。

### chunk size 的可归因实验

固定模型、权重/attention backend、GPU、并发到达轨迹、prompt/output 长度、KV 容量和随机种子，只改变 chunk size，例如 256/512/1024/2048。每档预热并重复，记录：

```text
请求级：queue time、TTFT、ITL/TPOT、E2E p50/p95/p99
系统级：input/output tok/s、iteration 时长、batch tokens、preemption
设备级：GEMM shapes、kernel launch 数、GPU active、activation 峰值
```

若小 chunk 改善 p99 ITL，却增加 iteration/launch 并降低 input tok/s，这是机制预期内的权衡，不应只挑一个指标宣称全面优化。

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
