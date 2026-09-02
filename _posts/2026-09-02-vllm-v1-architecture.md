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

### Offline 调用链：同步 API 不等于串行执行

按 v0.20.0 的职责边界，可以这样读 `LLM.generate()`：

```text
用户 Python
  → LLM.generate(prompts, sampling_params)
  → 输入预处理 / request ids
  → LLMEngine.add_request(...)
  → while unfinished: LLMEngine.step()
  → EngineCoreClient.get_output()
       ├─ InprocClient: 同进程驱动 EngineCore
       └─ MP client: 经 IPC 读取后台 EngineCore 的输出
  → RequestOutput 列表
```

`generate()` 对调用者是阻塞函数，但内部可以把 prompts 同时加入 engine，并在 iteration 边界动态组成 batch。它返回的是完成后的 Python 对象，不含 HTTP、SSE、客户端断连和网络 backpressure。

### Online 调用链：同一 core 外面多了服务生命周期

```text
HTTP client
  → FastAPI route / OpenAI 协议校验
  → chat messages --chat template--> prompt text/token ids
  → tokenizer + 参数校验
  → AsyncLLM / EngineCoreClient.add_request
  → 后台 EngineCore schedule→execute→update
  → EngineCoreOutput
  → OutputProcessor / detokenize / stop strings
  → SSE chunk 或最终 JSON
```

chat template 是把 `{role, content}` 消息转换成模型训练时约定的文本格式；tokenizer 再把文本变成 token ids。两份肉眼相同的消息若模板、special tokens 或 tokenizer 版本不同，得到的 token prefix 也可能不同，从而影响长度、logits 与 APC 命中。

健康检查、一次 HTTP 200、输出语义正确与负载性能是四种不同证据：

| 检查 | 能证明 | 不能证明 |
|---|---|---|
| `/health` | 前端进程可响应 | 模型 forward 成功 |
| 一次 200 | 完整路径至少成功一次 | 并发稳定、p99 达标 |
| 固定输入比对 | 当前配置下输出/分布符合预期 | 高吞吐 |
| 压测 + profiler | 指定 workload 的性能 | 所有模型与硬件通用 |

## 3. async task、进程和 GPU worker 不要混为一谈

- async task 是事件循环里的协作任务；`await` 在 I/O 等待时让出执行权；
- 进程拥有独立地址空间和 PID，跨进程需要 IPC；
- GPU worker 是某个 device/rank 的模型执行角色；
- rank 是分布式进程在全局或并行组中的编号，不是 HTTP request id。

v0.20.0 的 `EngineCoreClient.make_client` 会依据配置选择同进程或多进程客户端。`InprocClient` 可在当前进程调用 core；MP client 则通过后台 core 和 IPC 交换小型消息。共同接口让上层不必针对每种拓扑重写 add request/get output。

完整进程图应把“并发单位”和“模型分片单位”分开：

```text
server process
  ├─ HTTP event loop
  │    ├─ async task: request A
  │    ├─ async task: request B
  │    └─ async task: streaming writer
  └─ EngineCoreClient
        │ inproc call 或 IPC
        ▼
EngineCore process
  └─ executor
       ├─ worker process / global rank 0 → GPU 0
       ├─ worker process / global rank 1 → GPU 1
       └─ ... TP/PP/DP collective groups
```

`await` 只表示当前协程在等待时让出事件循环，不会凭空创建 OS process，也不会让同一 GPU 同时无限执行 kernels。每个 worker 内还可能有 CPU 准备、CUDA streams、CUDA Graph 和 device kernels 的并行/重叠；这与“有几个 worker”是不同层次。

常见 client 选择可归纳为：

| client | core 所在位置 | 上层看到的接口 | 主要边界 |
|---|---|---|---|
| `InprocClient` | 同进程 | add/abort/get output | Python 直接调用 |
| `SyncMPClient` | 后台进程 | 同步调用 | IPC 阻塞边界 |
| `AsyncMPClient` | 后台进程 | async 调用 | event loop + IPC |

DP、外部 launcher 或其他 executor 可能增加子类/进程，表格不是所有拓扑枚举。准确表述应是 `LLMEngine.step → EngineCoreClient.get_output`，然后才根据 client 类型落到同进程 core 或 IPC；不能无条件画成 `LLMEngine.step → EngineCore.step`。

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

### 每种对象的输入、输出与所有者

| 对象 | 创建者 | 所有者/修改者 | 典型字段 | 生命周期 |
|---|---|---|---|---|
| HTTP schema | route | 前端 | messages、model、stream | 一次网络请求 |
| `EngineCoreRequest` | 前端/engine facade | 作为消息传给 core | prompt ids、sampling params、request id | 跨边界传输 |
| `Request` | `Request.from_engine_core_request` | Scheduler | status、all ids、computed、spec ids | 整个推理请求 |
| `SchedulerOutput` | `Scheduler.schedule()` | executor 消费 | scheduled counts、new/cached/finished、block ids | 一次 iteration |
| `ModelRunnerOutput` | model runner | Scheduler 消费 | sampled ids、logprobs 等 | 一次 iteration |
| `EngineCoreOutput` | Scheduler/core | 前端消费 | 新 token ids、finish reason | 一次更新 |
| `RequestOutput` | output processor | API 调用者 | 文本、token、metrics | 对外返回 |

`ModelRunnerOutput.sampled_token_ids` 跨进程时适合用小型 Python lists；这不表示 sampling 必然在 CPU。设备上可以先产生 tensor，再在必须推进控制状态的边界做 copy/转换。反过来，把巨大 K/V payload 放进每轮 IPC 消息会引入地址空间、同步和传输问题，所以消息通常传 block ids，而 K/V 常驻 worker 的 GPU pool。

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

“基于真实输出的 token/stop 更新必须在 execute 之后”是一个正确性约束：schedule 只承诺准备算多少 token；真正的 sampled ids、推测解码接受长度、模型错误和部分停止条件要等执行结果。要注意 v0.20.0 的 Scheduler 会在计划构造后先把 scheduled positions 乐观计入 `num_computed_tokens`，把它当作“已完成或已提交的调度前沿”；这不是提前伪造 sampled token。失败、拒绝或异步差异仍要在 output 处理时校正。

```python
# 教学伪代码：同步核心主干
while has_work():
    plan = scheduler.schedule()              # 读权威状态，预留资源
    try:
        runner_output = executor.execute(plan)
    except Exception:
        scheduler.rollback(plan)             # 不能保留虚假的进度
        raise
    core_outputs = scheduler.update_from_output(plan, runner_output)
    publish(core_outputs)
```

输入是 Scheduler 的长期请求状态和资源状态；输出是本轮可发布的新 token。`schedule()` 可能修改预留/队列元数据，`update` 才按真实结果提交 token 进度。真实 v0.20.0 还处理 grammar bitmask、执行期间 abort、KV connector、错误诊断、pipeline/batch queue 等分支；教学伪代码不能证明某个运行配置走同步路径。

### 为什么先初始化真实 KV 容量，再构造依赖它的调度器

启动主线可以概括为：

```text
create ModelExecutor
  → get_kv_cache_specs()
  → determine_available_memory()
  → get_kv_cache_configs(...)
  → generate scheduler KV-cache config
  → executor.initialize_from_config(...)  # worker 建真实 GPU cache
  → create Scheduler(config)               # 管同一容量的控制面 blocks
```

Scheduler 判断能否准入/增长请求，必须先知道 pool 有多少 blocks、block size 与 cache groups。控制面 `KVCacheManager` 保存 block ids、hash、ref count 等；worker 保存 K/V tensor。两者由 `SchedulerOutput` 同步映射。如果 Scheduler 认为请求拿到 physical block 7，而 worker-side `BlockTable` 没收到更新，kernel 就可能寻址错误。

无 KV 的模型、connector、elastic expert parallel、PP batch queue 等有条件分支；上述流程只表达资源依赖，不把所有配置写成同一条无条件路径。

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

### block table 从 Python metadata 到 device tensor

```text
Scheduler / KVCacheManager
  request A → physical block ids [7, 3, 11]
          │ SchedulerOutput（小型控制消息）
          ▼
GPUModelRunner._update_states
  更新 worker-side request state / BlockTable
          │ commit / copy
          ▼
device block_table tensor [num_reqs, max_blocks_per_req]
          │ attention metadata
          ▼
backend kernel: logical position → block id → slot → KV pool address
```

概念布局可写成 `kv_cache[layer][K_or_V][physical_block][offset][kv_head][head_dim]`，但真实轴顺序、是否 K/V 合并、page/block 大小和 dtype 必须查选中的 cache spec/backend。这张图只说明地址翻译，不是 ABI。

CPU 适合不规则、分支多的请求队列和引用状态；GPU 适合批量 tensor 计算。GPU 并非“总是更快”：把 Scheduler 的大量 Python 分支改成许多微小 kernels 可能增加 launch/sync，而每步从 CPU 经 PCIe 搬权重/KV 又会让传输成为瓶颈。

### 一次请求的端到端时序

```text
client      front end       core/scheduler       runner/GPU
  | POST       |                   |                  |
  |----------->| template/tokenize |                  |
  |            |---add request---->| WAITING          |
  |            |                   |--schedule------->|
  |            |                   |  prompt + blocks | forward/sample 21
  |            |                   |<--runner output--|
  |            |                   | update computed=4|
  |<--SSE 21---|<--core output-----|                  |
  |            |                   |--schedule 21---->|
  |            |                   |<--sample 22------|
  |<--SSE 22---|<--update----------|                  |
  |            |                   |--schedule 22---->|
  |            |                   |<--sample 23------|
  |<--SSE 23---|<--finish/free-----|                  |
```

客户端断连不等于 core 请求自动消失。前端要检测断连并发出 abort/cancel；若请求已在执行，本轮仍可能完成后才观察到取消。server TTFT 可包含网络、解析和排队，而 Offline TTFT 没有同一边界，比较时必须声明计时起点。

## 8. 性能问题应回到所属层

- `num_scheduled_tokens` 不合理：先查 Scheduler；
- block id/ref count 异常：查 KVCacheManager/BlockPool；
- slot mapping 或 packed shape 错：查 GPUModelRunner/BlockTable；
- kernel/dtype/layout 不支持：查 attention backend；
- HTTP 200 但内容异常：还要检查模板、采样、finish reason 与 detokenization；
- GPU 间出现大段空隙：区分 queue、CPU preparation、IPC、launch gap 和 kernel device time。

### 推荐断点与日志字段

| 目标 | v0.20.0 阅读/断点入口 | 建议记录 |
|---|---|---|
| 外部请求形成 | `entrypoints/llm.py::LLM.generate` 或 OpenAI route | request id、prompt tokens、sampling params |
| client 拓扑 | `v1/engine/core_client.py::make_client` | client class、MP/DP 配置 |
| core 迭代 | `v1/engine/core.py` 的 step 主干 | iteration、schedule/execute/update 时间 |
| 调度计划 | `v1/core/sched/scheduler.py::schedule` | waiting/running、budget、per-request scheduled tokens |
| KV 分配 | `v1/core/kv_cache_manager.py::allocate_slots` | blocks、hits、free blocks、preemption |
| worker 同步 | `v1/worker/gpu_model_runner.py::_update_states` | new/cached/finished requests、block ids |
| device 输入 | runner input/metadata builder | input ids、positions、query starts、block table shape |
| backend | `v1/attention/backends/registry.py` 与选中 backend | backend 名、dtype、layout、block size |

这些断点分别证明静态调用或某次动态路径。类存在于注册表只证明它是候选；必须看到启动日志/trace 才能说本次运行选中了它，必须有 profiler 才能归因性能。

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
