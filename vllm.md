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
