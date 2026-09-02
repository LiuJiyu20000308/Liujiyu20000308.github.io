---
layout: page
title: vLLM：从 KV Cache 到高吞吐推理服务
permalink: /vllm/
lang: zh-CN
math: true
---

这组文章面向已经理解 Transformer 自注意力、但刚开始阅读 vLLM 源码的读者。它不按目录罗列类名，而是从推理服务的三本成本账出发：每个 token 要做多少计算，历史状态怎样占用和流动，多个长度不同的请求怎样共享有限的 GPU 时间与显存。

## 总体知识图

```text
Transformer 自回归生成
        │
        ├─ 重算历史太贵 ───────────→ KV Cache
        │                                │
        │                                ├─ 容量与并行公式
        │                                ├─ 分页分配 → PagedAttention
        │                                └─ 跨请求复用 → Prefix Caching
        │
        ├─ 请求长度和到达时间不同 ──→ Continuous Batching
        │                                ├─ Scheduler / token budget
        │                                └─ Chunked Prefill
        │
        ├─ 单个执行步骤仍有成本 ─────→ 量化 / 推测解码
        │
        └─ Attention 中间结果搬运昂贵 → FlashAttention

上述机制由 vLLM V1 的前端、EngineCore、Scheduler、KVCacheManager、
ModelExecutor、GPUModelRunner 与 attention backend 串成一次完整请求。
```

分页、缓存、调度和 kernel 不是互斥菜单。PagedAttention 组织 KV 的物理存储，Automatic Prefix Caching 复用兼容的完整前缀块，Scheduler 决定本轮推进哪些 token，FlashAttention 等 backend 执行具体注意力计算。只有把层次分开，才能解释为什么“功能已开启”不等于“端到端一定更快”。

## 推荐阅读顺序

1. [KV Cache：从历史重算到容量公式]({{ '/vllm/kv-cache/' | relative_url }})  
   先建立 prefill、decode、K/V shape 和单 token 容量的共同语言。

2. [vLLM V1 架构：请求怎样穿过推理服务]({{ '/vllm/architecture/' | relative_url }})  
   区分 HTTP 前端、核心循环、控制面消息和 GPU tensor。

3. [PagedAttention：从显存碎片到 slot mapping]({{ '/vllm/paged-attention/' | relative_url }})  
   沿 logical block、physical block、BlockTable 和 slot 看分页 KV 的完整寻址链。

4. [Automatic Prefix Caching：从 hash chain 到物理块复用]({{ '/vllm/prefix-caching/' | relative_url }})  
   解释什么前缀可以安全复用，以及命中率为什么不能直接换算成延迟收益。

5. [Scheduler：Continuous Batching 与 Chunked Prefill]({{ '/vllm/scheduler/' | relative_url }})  
   用逐轮时间线理解 token budget、请求状态、抢占和长 prompt 切块。

6. [量化与推测解码：怎样减少每轮成本]({{ '/vllm/quantization-speculative-decoding/' | relative_url }})  
   分清权重、激活、KV 与累加器，再理解 proposer/target 的分布修正。

7. [FlashAttention：从 IO-aware tiling 到 Hopper 异步流水]({{ '/vllm/flashattention/' | relative_url }})  
   从 online softmax 走到 FlashAttention v1、v2、v3，并与近似 attention 区分。

## 七篇教程的详细目录

### 1. KV Cache：先把每个 token 的状态账算对

- hidden states 如何经 $W_Q/W_K/W_V$ 或 fused QKV projection 产生 Q/K/V；
- MHA、GQA、MQA 的 tensor shape 与 RoPE 的作用位置；
- prefill/decode 时间线、$P+G-1$ 停止边界、8 vs 30 与 42 vs 190；
- 每 token/请求/batch/rank 的容量公式和可执行计算器；
- TP head 分片/复制、PP 层切分、DP replica，以及 block-rounded/pool 三种容量；
- beam、speculative、sliding window、APC、offload、KV transfer 的边界。

### 2. V1 架构：从 API 到 GPU 的对象变换

- Offline `LLM.generate()` 与 OpenAI-compatible server 两条调用链；
- async task、OS process、worker、global/local/TP rank 的区别；
- `EngineCoreRequest → Request → SchedulerOutput → ModelRunnerOutput → RequestOutput`；
- EngineCore、Scheduler、KVCacheManager、executor、worker、runner、backend 的所有权；
- `[11,12,13,14] → [21,22,23]` 的逐轮状态和 computed=6/all=7；
- block ids 怎样从 CPU metadata 提交为 device table，以及推荐断点/日志。

### 3. PagedAttention：从碎片到 kernel 地址

- internal/external fragmentation、block size 取舍和容量反例；
- logical block、physical block、offset、slot 的公式与边界手算；
- 含原子失败检查和 asserts 的最小 allocator；
- `KVCacheBlock`、null block、ref count、intrusive free queue 与 eviction；
- Scheduler→GPUModelRunner→worker BlockTable→backend metadata 的传播；
- allocation/kernel block size、cache groups、parallel config 与证据层级。

### 4. Automatic Prefix Caching：精确身份与生命周期

- prefix、substring、语义相似为何不同；模板/tokenizer 如何改变 token prefix；
- parent hash chain、full/partial block、salt/LoRA/多模态/group identity；
- Request 产生 hash、GPU 写完 KV、BlockPool 提交之间的时序；
- hash→多个 physical 副本与 ref=2 共享一份的区别；
- `[7,19,23]` 命中/分配时间线、touch/free/evict 状态机；
- 1712/1729 counter delta、TTFT/ITL/E2E 与可归因实验。

### 5. Scheduler：把“阶段”还原成统一进度

- client/static/iteration batch 与 schedule 到达边界；
- WAITING/RUNNING/PREEMPTED/FINISHED 不等于 prefill/decode；
- logical/spec/placeholder/computed 的 deficit 与乐观已提交进度；
- token budget、sequence slots、KV blocks 三种资源；
- A/B/C 五轮 19-token 守恒和可执行 simulator；
- RUNNING/WAITING 主干、victim/rollback/recompute、APC/remote KV；
- 6000-token Chunked Prefill、full-ISL reserve 与 chunk size 实验。

### 6. 量化与推测解码：两条独立成本轴

- bit/byte/编码数、affine quantization 的完整推导与分组代码；
- 权重/激活/累加器/KV，W4A16 与运行时片上反量化；
- INT4/INT8/FP8 E4M3/E5M2 的表示和 roofline/Amdahl 边界；
- AWQ/GPTQ 量化层与 Marlin 执行层的职责图；
- KV dtype/scale/backend 能力链和容量计算器；
- proposer/target 的 p/q 获取、accepted/recovered/bonus；
- 正残差证明、精确枚举、Monte Carlo 与单变量实验矩阵。

### 7. FlashAttention：完整 attention 的 IO 优化

- FLOPs 与 FLOPS、HBM/SRAM/register 和普通 attention IO 时间线；
- tiling 与 online softmax 推导、可执行数值对照；
- v1 forward/backward、LSE、$D_i=dO_i\cdot O_i$ 和重计算；
- v1 sliced-K、v2 sliced-Q、Q-sequence parallelism 与 SM 利用率；
- MEA、Longformer、Performer/FAVOR+ 的 exact/sparse/approximate 区别；
- v3 TMA/WGMMA、producer-consumer、ping-pong/barrier；
- FP8 block quantization、Hadamard、精度数据流和转换成本；
- vLLM backend 从候选到 profiler 的证据链。

## 完整知识依赖图

```text
Transformer hidden state
  └─ Q/K/V projection + RoPE
       └─ 自回归 KV Cache
            ├─ 容量公式 ── TP/PP/DP 与 dtype
            ├─ 物理存放 ── PagedAttention
            │                 ├─ BlockPool / BlockTable / slot mapping
            │                 └─ ref count + hash ── APC
            ├─ 每轮增长 ── Scheduler
            │                 ├─ Continuous Batching
            │                 ├─ Chunked Prefill
            │                 ├─ preemption/recompute
            │                 └─ speculative token budget
            ├─ 每元素 byte ── KV quantization
            └─ 每步 attention IO ── FlashAttention/backend

HTTP/chat template/tokenizer
  → EngineCoreRequest
  → EngineCore/Scheduler（CPU 控制面）
  → SchedulerOutput / block ids
  → GPUModelRunner（打包 tensors/metadata）
  → model layer / attention backend（GPU 数据面）
  → sampled ids → detokenize/SSE
```

建议按依赖从上往下读。若直接从 FlashAttention kernel 开始，很容易不知道 block table 从哪里来；若只读 Scheduler，又会把 token budget 与显存 byte 混为一谈。

## 根据问题查文章

| 我的问题 | 先读 | 重点章节 |
|---|---|---|
| K/V 是从哪里算出来的？ | KV Cache | QKV projection、shape、RoPE |
| 为什么生成 5 个 token 只回灌 4 个？ | KV Cache | prefill/decode 与停止边界 |
| TP=16 为什么 KV 没继续减半？ | KV Cache | rank-local heads 与复制 |
| HTTP 请求最后怎样变成 GPU tensor？ | V1 架构 | 两条调用链与对象生命周期 |
| async、process、worker、rank 有什么区别？ | V1 架构 | 进程边界图 |
| block table `[7,3,11]` 怎么读？ | PagedAttention | logical→physical→slot |
| slot 是不是一个 K/V？ | PagedAttention | slot payload 与 cache spec |
| `ref=0` 为什么还能命中？ | PagedAttention/APC | free queue 与 cached 候选 |
| APC 命中 99%，为什么速度没快 99%？ | APC | counter delta 与指标拆分 |
| 开头一个 token 不同，后面为何全 miss？ | APC | parent hash chain |
| `num_computed_tokens` 为何在 GPU 完成前增加？ | Scheduler | 已提交进度与 output 校正 |
| placeholder 会不会让 EOS 后多算？ | Scheduler | async placeholder |
| Chunked Prefill 会减少总 FLOPs 吗？ | Scheduler | 6000-token 时间线 |
| INT4 权重是否先完整还原成 BF16？ | 量化/推测解码 | W4A16 kernel 数据流 |
| KV pool 启动已分配，为何请求还要 allocate？ | 量化/分页 | 物理 pool 与逻辑 ownership |
| 第三个 draft 被拒后从哪里取新 token？ | 量化/推测解码 | 正残差分布 |
| FlashAttention 为什么仍是 $N^2$？ | FlashAttention | exact tiling 与 IO |
| v1 已并行，v2 还优化什么？ | FlashAttention | sliced-K→sliced-Q |
| Q/K 同时 Hadamard 为何保持分数？ | FlashAttention | $QRR^TK^T$ 推导 |

## 术语索引

| 术语 | 本系列中的精确定义 |
|---|---|
| prefill | 并行处理尚未计算的输入前缀并建立各层 KV |
| decode | 用新 token 与已有 KV 继续 next-token prediction |
| TTFT | 请求起点到首个输出 token 的时间；起点必须声明 |
| ITL/TPOT | 连续输出 token 之间/每输出 token 的延迟 |
| block | 固定数量 token slots 的分配/布局单位，需说明层次 |
| slot | 一个 token 在 physical block 内的缓存位置，不是单个标量 |
| block table | 每请求 logical block index 到 physical block id 的映射 |
| KV pool | 引擎启动时预留的物理 cache 存储；不同 replica 通常独立 |
| APC | 跨兼容请求复用精确 full-prefix KV blocks |
| token budget | 单次 iteration 最多安排的 token positions，不是输出上限 |
| rank | 分布式 worker 进程在全局或并行组中的编号 |
| backend | 定义 attention metadata/layout/capability 并执行具体路径的实现 |
| exact attention | 保留完整连接/数学目标；仍会有浮点舍入 |
| registered candidate | 源码中可发现的实现，不代表本次运行已选择 |

## 证据标记与阅读约定

文章把结论分成六层：通用数学/算法、论文描述、vLLM v0.20.0 静态源码、注册候选、启动日志/trace 的动态路径、profiler/benchmark 的性能结果。出现“可能”“理论 payload”“教学伪代码”不是含糊，而是在阻止跨层推理：

```text
源码有实现 ≠ 当前配置选中
当前配置选中 ≠ 这次真的执行
这次执行过 ≠ 它是性能瓶颈
容量减半 ≠ 延迟减半 ≠ 吞吐翻倍
```

## 阅读时始终追问四件事

- 当前说的是通用算法、vLLM v0.20.0 的静态代码，还是一次真实运行？
- 优化改变的是 FLOPs、HBM 流量、显存容量，还是调度等待？
- 数据现在位于 Python 控制面、进程间消息，还是 GPU tensor？
- 指标是 TTFT、ITL、端到端延迟、吞吐、容量还是输出质量？

这套文章以 vLLM v0.20.0 为主要源码基线。版本相关的类名和路径均按该版本表述；算法结论与实测结论则分别标注，避免用一次实验替代通用规律。

## 主要资料

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691) 与 [FlashAttention-3](https://arxiv.org/abs/2407.08608)
