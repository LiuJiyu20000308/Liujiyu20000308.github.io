---
layout: post
title: "vLLM Automatic Prefix Caching：从 hash chain 到物理块复用"
date: 2026-09-02 09:30 +0800
tags: [vLLM, LLM 推理, Automatic Prefix Caching, KV Cache, 哈希]
toc: true
math: true
permalink: /vllm/prefix-caching/
---

## 本篇要回答什么

普通 KV Cache 只让同一请求在后续 decode step 复用自己的历史。Automatic Prefix Caching（APC，自动前缀缓存）更进一步：新请求若拥有兼容的相同 token 前缀，可以直接引用已经算好的物理 KV blocks，跳过命中部分的 prefill。

APC 的关键不是“两个问题意思相似”，而是**从 position 0 开始的模型输入 token、父前缀身份和影响 K/V 的额外条件全部兼容**。

## 1. prefix 不等于 substring 或语义相似

序列 A、B 有长度 $m$ 的共同 prefix，表示

$$
a_i=b_i,\qquad i=0,\ldots,m-1.
$$

相同 substring 只要求中间某段一致。对 causal decoder，同样的 token block 位于不同历史之后时，高层 hidden states 和 K/V 通常不同，不能直接拼接。

```text
A: [system][document][question A]
B: [system][document][question B]  → 前两部分可能复用

C: [dynamic id][system][document] → 第一个 block 已不同，后续 hash chain 也不同
```

APC 比较 chat template 和 tokenizer 处理后的 token ids，不比较 UI 中看起来相似的字符串。标点、special tokens、工具 JSON 顺序和模板版本都可能改变前缀。

## 2. 为什么必须包含 parent hash

设 block size 为 4：

```text
block 1: [A B C D]
block 2: [E F G H]
block 3: [I J K L]
```

链式身份可概括为：

$$
H_1=hash(tokens_1, extra),
$$

$$
H_2=hash(H_1,tokens_2,extra),
$$

$$
H_3=hash(H_2,tokens_3,extra).
$$

如果只 hash 当前 block 的 `[E F G H]`，它出现在不同父历史后也会被误判为同一状态。parent hash 把此前所有完整 blocks 纳入当前身份。

RoPE 等位置编码还意味着同一个 token 位于 position 10 与 100 时 K/V 不同。位置在普通连续前缀中由父链和 block 顺序隐含；任意中间片段匹配则缺少这一保证。

## 3. 为什么只缓存 full blocks

完整 block 的 token 内容不会再增长，hash 身份稳定。partial block 之后追加 token 会改变其内容：

```text
[I J _ _] → [I J K _] → [I J K L]
```

因此普通 APC 以完整 blocks 为稳定复用单位。命中长度常是 block size 的整数倍；尾部不足一块的公共 token 仍可能需要计算。

“生成了 hash”也不等于“缓存命中”：

```text
hash identity 存在
        +
cache map 中仍有有效 physical block
        +
额外身份条件兼容
        =
可复用的 cache hit
```

## 4. 额外身份条件

token ids 相同仍可能不能共享。实现需要考虑会改变 K/V 的上下文，例如：

- `cache_salt`：主动隔离缓存域；
- LoRA/adapter 身份：权重不同会产生不同 K/V；
- 多模态输入及其 hash；
- attention/cache group 身份；
- 模型或服务生命周期中的兼容性边界。

因此 APC 不是把 `hash(token_ids)` 当成唯一键。安全性优先于多命中几个 blocks。

## 5. 从请求 hash 到物理块复用

假设 cache 已保留：

```text
H1 → physical block 7,  ref=0
H2 → physical block 19, ref=0
```

新请求 R 为：

```text
[A B C D | E F G H | I J]
```

完整流程是：

1. 请求创建阶段为前两个 full token blocks 计算 `H1`、`H2`；
2. Scheduler 通过 KVCacheManager 查询最长连续命中，得到 8 个 computed tokens 和 blocks `[7,19]`；
3. `touch` 让命中块从 ref=0 变为 ref=1，并从 free queue 移出；
4. 为未命中的 `[I J]` 分配新 block 23；
5. 请求的 block table 成为 `[7,19,23]`；
6. worker 读取 7、19 的历史 K/V，只为 `[I J]` 写入 23；
7. 请求结束时 ref count 下降，ref=0 的 cached blocks 回到驱逐候选队列，但内容和 hash 可暂时保留。

```text
token ids
   ↓
BlockHash：内容与父前缀身份
   ↓ lookup
physical block ids：GPU 存储位置
   ↓
BlockTable：请求对这些块的引用
   ↓
runner 读取命中 KV / 写入未命中后缀
```

当 block 7 要被新内容覆盖时，必须先删除旧 `H1 → 7` 映射并重置身份，否则未来会错误命中已被覆盖的数据。

## 6. `ref_cnt=0` 为什么仍可能是缓存

请求所有权与缓存内容有效性是两个维度：

| 状态 | 含义 |
|---|---|
| `ref_cnt>0` | 至少一个活跃请求正在使用，不可驱逐 |
| `ref_cnt=0` 且有 hash | 无活跃 owner，但仍可快速命中；容量紧张时可驱逐 |
| `ref_cnt=0` 且无 hash | 普通空闲 block |

这种设计让 APC 无需把所有历史内容永久锁在显存中：热前缀可以再次被 touch，冷前缀则为新请求让路。

## 7. 一次本机实验应怎样解释

一次保留的实验环境为 RTX 5070 Ti、Qwen3-0.6B、vLLM v0.20.0，开启 APC 与 eager execution。两个请求共享长前缀，只改变末尾问题：

| 请求 | hits 增量 | queries 增量 | 单请求 hit rate | E2E 墙钟 |
|---|---:|---:|---:|---:|
| 第一次 | 0 | 1727 | 0% | 0.182 s |
| 第二次 | 1712 | 1729 | 99.02% | 0.178 s |

counter 是累计量，单请求必须计算

$$
\text{hit rate}=\frac{\Delta hits}{\Delta queries}
=\frac{1712}{1729}=99.02\%.
$$

$1712=107\times16$，与该次运行观察到的 block size=16 和 full-block 命中一致。但“99% 输入命中”不等于“端到端加速 99%”：APC 只省命中 prefix 的 prefill，墙钟还包含 HTTP、queue、未命中输入、decode、sampling、CPU 和测量噪声。每个条件只有一个样本，所以 4 ms 差异不足以估计稳定性能收益。

这组数据的强结论是缓存确实命中 1712 个输入 tokens；弱结论是当前单样本没有分辨出明显 E2E 差异。

## 8. APC 更可能改善哪些指标

APC 主要减少重复 prefill，通常优先影响：

- TTFT（Time To First Token）；
- input token throughput；
- 重复前缀计算和存储。

它不会免除每个 decode step，也不会让新 query 停止读取历史 KV，所以 ITL（Inter-Token Latency）未必改善。输出越长，decode 越容易稀释 prefill 节省。

可复现实验应扫描：

```text
公共前缀：128 / 512 / 2048 tokens
输出长度：1 / 32 / 256 tokens
APC：      off / on（分别重启清状态）
干扰：     无 / 插入大量其他 prefixes 触发 eviction
```

每组 warm up 后重复足够样本，报告 median/P95/P99，并同时记录 prefix counters、TTFT、ITL、E2E 与吞吐。还应加入三个反例：开头一个 token 不同、相同 tokens 但 salt 不同、命中前发生驱逐。

## 常见误区

- “意思相同就能复用”：APC 是精确 token prefix caching，不是 embedding 检索。
- “hash 相同就一定命中”：还要有有效物理块和兼容身份。
- “hit rate 80%，总 KV 容量减少 80%”：不同后缀、decode、新 partial block 仍各自增长。
- “APC 加速 decode”：它主要跳过命中前缀的 prefill。

## 源码阅读入口（v0.20.0）

- `vllm/v1/core/kv_cache_utils.py`：block hash、BlockPool 和 cached blocks；
- `vllm/v1/core/kv_cache_manager.py`：查找、分配、touch/free；
- `vllm/v1/core/sched/scheduler.py`：等待请求怎样接入 cached progress；
- `vllm/v1/request.py`：请求的 token 进度与 block hashes。

## 本篇总结

APC 将“新请求有相同开头”落实为一条严格的控制链：相同模型输入和额外身份产生相同 full-block hash chain，cache lookup 找到仍有效的 physical blocks，请求通过 ref count 引用它们，只计算未命中后缀。命中计数证明复用，性能收益仍必须由分离 TTFT、ITL 和吞吐的对照实验决定。

---

[上一篇：PagedAttention]({{ '/vllm/paged-attention/' | relative_url }}) · [系列首页]({{ '/vllm/' | relative_url }}) · [下一篇：Scheduler]({{ '/vllm/scheduler/' | relative_url }})

## 资料

- [vLLM Automatic Prefix Caching 文档](https://docs.vllm.ai/en/latest/design/v1/prefix_caching/)
- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
