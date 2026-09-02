---
layout: post
title: "vLLM PagedAttention：从显存碎片到 slot mapping"
date: 2026-09-02 09:20 +0800
tags: [vLLM, LLM 推理, PagedAttention, KV Cache, 显存管理]
toc: true
math: true
permalink: /vllm/paged-attention/
---

## 本篇要回答什么

请求的 KV Cache 会随生成逐 token 增长。如果每个请求必须占据一段连续显存，就会遇到最大长度预留、扩容、搬移和碎片问题。PagedAttention 的核心不是修改 attention 数学公式，而是让请求的逻辑连续 token 映射到不连续的物理 KV blocks。

## 1. 连续分配为什么困难

若 A、B 的最大长度均预留 16 slots，实际只使用 5 和 12：

```text
A [#####...........]  allocated=16, used=5
B [############....]  allocated=16, used=12
```

分配 32，实际有效 17，浪费的 15 位于已经分给请求的区域内部，这是 internal fragmentation（内部碎片）。

改成按需连续分配仍有 external fragmentation（外部碎片）：

```text
初始：[AAAAAAAA][BBBBBB][CCCCCC]
释放 A/C 后：[........][BBBBBB][......]
```

总空闲为 14，最大连续洞只有 8；需要连续 10 slots 的 D 仍无法进入，除非搬移 B。

## 2. 固定大小 block 缩小连续性要求

将 KV pool 切成 block size 为 $B$ 的 physical blocks。长度为 $S$ 的请求需要

$$
n=\left\lceil\frac{S}{B}\right\rceil
$$

个 blocks，末块浪费

$$
w=nB-S,\qquad 0\le w<B.
$$

分页没有消除全部内部碎片，但把每请求浪费限制到末块的至多 $B-1$ slots，并消除“整个请求必须物理连续”的要求。block 越小，尾部浪费和 prefix 复用粒度越细；代价是更长的 block table、更多 metadata 和寻址工作。

## 3. logical block、physical block 与 slot

设 block size 为 4，请求有 10 个 tokens，block table 为 `[7,3,11]`。读取 position 9：

$$
b_{logical}=\left\lfloor\frac{9}{4}\right\rfloor=2,
\qquad offset=9\bmod4=1.
$$

查 block table：

```text
logical block 0 → physical block 7
logical block 1 → physical block 3
logical block 2 → physical block 11
```

所以 position 9 位于 physical block 11 的 offset 1。若只看一维 slot 编号：

$$
slot=11\times4+1=45.
$$

真实 backend 还要把 slot 解释为 `[layer, K/V, block, head, offset, head_dim]` 等具体 layout；45 不是 CUDA 指针，只是逻辑寻址结果。

## 4. 一个最小 allocator 时间线

block size=4，pool 有 blocks 0..5：

```text
add A=6   → A:[0,1],   free:[2,3,4,5]
add B=3   → B:[2],     free:[3,4,5]
free B    →             free:[3,4,5,2]
grow A+3  → A:[0,1,3], free:[4,5,2]
add C=5   → C:[4,5],   free:[2]
free A    →             free:[2,0,1,3]
add D=10  → D:[2,0,1], free:[3]
```

D 的三个 logical blocks 使用不连续的物理 blocks `[2,0,1]`，无须搬移 C。增长时必须先计算所需增量并检查容量，再原子地修改 table；容量不足时若先拿走一部分，会留下泄漏或半完成映射。

## 5. `KVCacheBlock` 不只是一个整数

教学 allocator 可以只保存 block id。vLLM v0.20.0 为共享和驱逐还维护控制面 metadata：

- `block_id`：物理块编号；
- `ref_cnt`：当前有多少 owner；
- `block_hash`：可选的缓存身份；
- free-list 前后指针：支持侵入式双向队列；
- `is_null`：特殊占位 block。

真实 K/V 数值仍位于 worker/GPU tensor，`KVCacheBlock` 本身不保存大体积缓存。

启用 prefix caching 时，free queue 中不全是“内容为空”的 blocks。一个 `ref_cnt=0` 的 cached block 可以保留 hash 和旧 KV：新请求命中时把它从 free queue 中 touch 出来；需要容量时又可以驱逐旧身份并覆盖。

```text
ref_cnt > 0：正在被请求引用，不可覆盖
ref_cnt = 0 且有 hash：可命中，也可作为驱逐候选
ref_cnt = 0 且无 hash：普通空闲 block
```

## 6. block id 怎样进入 attention backend

完整控制链可以压缩为：

```text
KVCacheManager.allocate_slots
       │ 返回 KVCacheBlock objects
       ▼
SchedulerOutput
  NewRequestData.block_ids / CachedRequestData.new_block_ids
       │
       ▼
GPUModelRunner._update_states
       │ 更新本地请求 row
       ▼
worker BlockTable
  CPU block-table buffer → commit 到 device
       │ positions + block size
       ▼
slot mapping / attention metadata
       │
       ▼
backend 读取历史 KV，并把新 K/V 写到物理 slots
```

cached request 每轮通常只传增量 block ids，因为 worker 保留已有状态。存在多个 KV cache groups 时，外层 tuple 对应不同 cache group，不能永远假定只有一组 full-attention cache。

## 7. PagedAttention 的收益链

PagedAttention 的直接收益是容量组织：

```text
减少最大长度预留和外部碎片
            ↓
同一 KV pool 可容纳更多有效 tokens/requests
            ↓
Scheduler 有机会形成更大、更持续的 iteration batch
            ↓
权重读取和 GPU 执行得到更好摊销
            ↓
serving throughput 可能提高
```

每个箭头都有条件。batch=1 的短请求若从未触及 KV 容量边界，分页不会凭空扩大 batch；metadata、block table 和间接寻址反而有成本。

新 query 仍计算

$$
\operatorname{softmax}(QK^T/\sqrt D)V
$$

并读取全部可见历史。分页改变存放与寻址，不减少数学上必须访问的 tokens。

## 8. 不要问“PagedAttention 还是 FlashAttention”

它们处在不同层次：

- PagedAttention：KV 的分页组织、共享和寻址；
- FlashAttention：exact attention 的 IO-aware 计算方法；
- Continuous Batching：本轮组合哪些请求；
- APC：哪些完整前缀 blocks 可跨请求复用；
- KV 量化：每个缓存元素使用多少 byte。

一个 backend 可以同时消费 paged KV layout，并使用 FlashAttention/FlashInfer 风格 kernel。

## 9. 三层证据

本地或网上的旧 `docs/design/paged_attention.md` 可能明确说明它描述原始 vLLM 论文，而非当前代码。阅读时区分：

1. 通用/论文思想：logical-to-physical blocks、按需分配、共享；
2. 历史 kernel 材料：早期 CUDA kernel 的线程、warp 和 layout；
3. 固定版本事实：v0.20.0 的 BlockPool、KVCacheManager、BlockTable、GPUModelRunner 与最终 backend。

`registry 中存在` 只证明候选实现可被发现；selector 返回和启动日志才能证明某配置选择了它；动态 trace 证明执行过；profiler 才能说明性能占比。

## 常见误区

- “分页后没有碎片”：仍有每请求末块浪费。
- “physical blocks 必须连续”：逻辑连续正是通过 block table 映射到不连续物理块。
- “block id 就是显存地址”：它仍需 backend layout 解释。
- “PagedAttention 将 attention 复杂度变成线性”：它管理 KV，不改变完整 causal attention 的数学依赖。

## 源码阅读入口（v0.20.0）

- `vllm/v1/core/kv_cache_utils.py`：`KVCacheBlock`、free queue 和 BlockPool；
- `vllm/v1/core/kv_cache_manager.py`：请求级 block 管理；
- `vllm/v1/worker/block_table.py`：worker 侧 table 与 slot mapping；
- `vllm/v1/worker/gpu_model_runner.py`：SchedulerOutput 到 device batch；
- `vllm/v1/attention/backends/registry.py`：backend 候选注册。

## 本篇总结

PagedAttention 把“为每个请求寻找一段可增长的连续显存”改成“为逻辑块分配任意空闲物理块”。它为高并发、快速回收和前缀共享提供了基础，但仍需 Scheduler、BlockPool、GPUModelRunner 和 backend 一起完成从控制面 block id 到真实 KV 读写的闭环。

---

[上一篇：vLLM V1 架构]({{ '/vllm/architecture/' | relative_url }}) · [系列首页]({{ '/vllm/' | relative_url }}) · [下一篇：Prefix Caching]({{ '/vllm/prefix-caching/' | relative_url }})

## 资料

- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM v0.20.0 源码](https://github.com/vllm-project/vllm/tree/v0.20.0)
