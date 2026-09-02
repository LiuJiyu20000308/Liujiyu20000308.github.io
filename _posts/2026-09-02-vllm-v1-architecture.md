---
layout: post
title: "vLLM V1 架构：一条请求怎样穿过推理服务"
date: 2026-09-02 09:10 +0800
tags: [vLLM, LLM 推理, EngineCore, GPUModelRunner, 服务架构]
toc: true
math: true
permalink: /vllm/architecture/
---

## 本篇要回答什么

一个 PyTorch 自回归循环已经能生成文本，vLLM 为什么还需要前端、EngineCore、Scheduler、KVCacheManager、ModelExecutor、GPUModelRunner 和 attention backend？答案在于推理服务处理的不是一个固定 tensor，而是持续到达、长度不同、随时完成或断开的请求集合。

## 1. 正确生成与高吞吐服务是两件事

假设 A、B、C 的 prompt/output 长度分别为 4/2、4/8、20/2：

- 串行执行让 C 的首 token 等待 A、B 全部结束；
- 静态 batch 可能把 prompt pad 到 20，并在 A、C 提前完成后留下空 row；
- iteration-level scheduling 在每次模型执行结束后重新选择工作，完成的请求退出，新请求可在下一轮加入；
- C 的长 prefill 还能拆成 chunks，避免一次占满整轮。

因此 vLLM 除模型 forward 外至少还要解决：KV 容量、动态调度、变长 GPU 执行，以及 HTTP、tokenizer、流式输出、指标与断连处理。

## 2. Offline 与在线入口

Offline API 由调用者直接拥有 engine：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
outputs = llm.generate(
    ["请用一句话解释 KV Cache。"],
    SamplingParams(temperature=0.0, max_tokens=32),
)
```

`LLM.generate()` 是同步接口，但 prompts 列表仍由 engine 调度，不等于逐 prompt 串行执行。v0.20.0 可从 `vllm/entrypoints/llm.py` 的 `LLM.generate` 向下阅读。

在线服务则多一层长期运行的网络前端：

```text
HTTP POST /v1/chat/completions
  → JSON 校验 / chat template / tokenizer
  → EngineCoreRequest
  → EngineCore + Scheduler
  → GPU execution + sampling
  → RequestOutput
  → JSON 或 SSE streaming
```

端口只是操作系统接收连接的编号，不对应一张 GPU。`/health` 成功说明前端能响应健康检查，不证明完整生成路径、负载下 p99 或输出正确性。

## 3. async task、进程和 GPU worker 不要混为一谈

- async task 是事件循环里的协作任务；`await` 在 I/O 等待时让出执行权；
- 进程拥有独立地址空间和 PID，跨进程需要 IPC；
- GPU worker 是某个 device/rank 的模型执行角色；
- rank 是分布式进程在全局或并行组中的编号，不是 HTTP request id。

v0.20.0 的 `EngineCoreClient.make_client` 会依据配置选择同进程或多进程客户端。`InprocClient` 可在当前进程调用 core；MP client 则通过后台 core 和 IPC 交换小型消息。共同接口让上层不必针对每种拓扑重写 add request/get output。

## 4. 一条请求的表示怎样变化

固定 prompt ids `[11,12,13,14]`，最多生成 `[21,22,23]`：

```text
用户文本 / chat messages
        │  template + tokenize + validate
        ▼
EngineCoreRequest
        │  Request.from_engine_core_request
        ▼
Request(status=WAITING,
        all_token_ids=[11,12,13,14],
        num_computed_tokens=0)
        │  Scheduler.add_request
        ▼
waiting/running queues + block metadata
        │  Scheduler.schedule
        ▼
SchedulerOutput
        │  ModelExecutor / Worker
        ▼
GPUModelRunner: packed input tensors + attention metadata
        │  model forward + sampling
        ▼
ModelRunnerOutput(sampled ids/logprobs)
        │  Scheduler.update_from_output
        ▼
EngineCoreOutput → detokenize → SSE/JSON
```

`EngineCoreRequest` 是跨边界消息，`Request` 是 Scheduler 持续修改的运行状态，`SchedulerOutput` 是单次 iteration 的执行计划。三者生命周期不同。

## 5. 为什么最终 `num_tokens=7`，computed 可能只有 6

prefill 处理 4 个 prompt tokens 后采样 21；随后：

| iteration 输入 | 采样输出 | 执行后 `num_computed_tokens` | `all_token_ids` 长度 |
|---|---|---:|---:|
| prompt 4 tokens | 21 | 4 | 5 |
| 21 | 22 | 5 | 6 |
| 22 | 23 | 6 | 7 |

请求在 23 返回后结束，23 没有再次进入模型。于是逻辑 token 总数为 7，真正经过模型的为 6。这不是少算了结果，而是 next-token prediction 的停止边界。

## 6. 核心组件的所有权边界

| 组件 | 主要拥有/产生什么 | 刻意不做什么 |
|---|---|---|
| 前端 | HTTP、模板、tokenize、streaming | 不决定 GPU 每轮公平性 |
| EngineCore | core 生命周期、命令、schedule→execute→update 循环 | 不实现具体 Qwen attention |
| Scheduler | 请求状态、队列、token budget、单轮计划 | 不执行 GPU kernel |
| KVCacheManager | block 查找、分配、引用与释放 | 不做 softmax |
| ModelExecutor/Worker | 分布式执行与设备 worker 编排 | 不解析 HTTP JSON |
| GPUModelRunner | 把变长计划压成 device batch，准备 metadata | 不决定服务策略 |
| Attention backend | cache layout、能力约束和 attention kernel | 不拥有全局请求队列 |

一次 iteration 的主干是：

```text
EngineCore
  ├─ Scheduler.schedule()        产生本轮 token/block 计划
  ├─ ModelExecutor.execute()     把计划交给 workers
  │    └─ GPUModelRunner         打包 tensors 并执行模型
  └─ Scheduler.update(...)       接收 sampled tokens，推进请求状态
```

## 7. 控制面与数据面

Scheduler 长期操作 request id、status、token counts、block ids、hash、ref count 和队列。这些对象小、不规则、分支多，适合 CPU 控制面。

模型 weights、KV pool、input ids/positions、hidden states、logits、workspace 与 graph buffers 是 GPU 数据面，需要稳定 dtype、shape 与 layout。

`block_table` 能同时存在两种表示：Scheduler 侧是 Python block-id lists；runner/backend 可把它提交成 device tensor。分类取决于当前表示和消费者，不取决于变量名。

```text
控制面消息只携带：request 42 使用 blocks [3, 8]

GPU 数据面长期保存：
KV pool[physical block 3]
KV pool[physical block 8]
```

大体积 K/V 不会跟随每个 `SchedulerOutput` 在进程之间来回复制。消息传递的是位置和本轮工作量。

## 8. 性能问题应回到所属层

- `num_scheduled_tokens` 不合理：先查 Scheduler；
- block id/ref count 异常：查 KVCacheManager/BlockPool；
- slot mapping 或 packed shape 错：查 GPUModelRunner/BlockTable；
- kernel/dtype/layout 不支持：查 attention backend；
- HTTP 200 但内容异常：还要检查模板、采样、finish reason 与 detokenization；
- GPU 间出现大段空隙：区分 queue、CPU preparation、IPC、launch gap 和 kernel device time。

## 常见误区

- “async 表示开了新进程”：异步任务与进程不是同一种并发。
- “一个请求对应一个 GPU worker”：请求可被 scheduler 组织进共享 iteration batch。
- “SchedulerOutput 里有 block id，所以包含 KV 数值”：block id 只是控制面地址身份。
- “vLLM 加速训练”：vLLM 主线是 inference/serving；训练系统可以调用它生成 rollout，但它不执行反向传播和 optimizer。

## 源码阅读入口（v0.20.0）

- `vllm/entrypoints/llm.py`：Offline API；
- `vllm/entrypoints/openai/api_server.py`：在线服务入口；
- `vllm/v1/engine/core_client.py`：同进程与多进程 client；
- `vllm/v1/engine/core.py`：核心循环；
- `vllm/v1/request.py`：运行时 Request；
- `vllm/v1/core/sched/scheduler.py`：调度状态和单轮计划；
- `vllm/v1/worker/gpu_model_runner.py`：device batch 与模型执行。

## 本篇总结

vLLM V1 的关键不是多套重复的“engine”，而是清晰的边界：前端把外部请求翻译成 core 消息，Scheduler 在 CPU 控制面决定单轮工作，KVCacheManager管理物理块，GPUModelRunner 将变长计划变成 tensors，backend 执行具体 kernel。沿同一请求的表示变化阅读源码，比按目录逐个背类名更可靠。

---

[上一篇：KV Cache]({{ '/vllm/kv-cache/' | relative_url }}) · [系列首页]({{ '/vllm/' | relative_url }}) · [下一篇：PagedAttention]({{ '/vllm/paged-attention/' | relative_url }})

## 资料

- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
- [vLLM 官方文档](https://docs.vllm.ai/)
