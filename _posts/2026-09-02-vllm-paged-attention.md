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

分页不会创造容量。8 个 blocks、$B=4$ 时，请求长度 `[1,4,5,8,9]` 分别需要 `[1,1,2,2,3]` 块，共 9 块，仍然装不下；尾部浪费分别是 `[3,0,3,0,3]` slots。若反过来为每个请求按最大长度 12 预留，则要 60 slots，而有效 tokens 只有 27。PagedAttention 改善的是分配粒度与连续性，不压缩 K/V 元素，也不替代 offload。

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

slot 也不是“一个 K 或一个 V 标量”。在常规 cache spec 中，它表示**某请求的一个 token 位置在一个物理块中的位置**；对某一层，该 slot 对应这个 token 的所有本地 KV heads：

```text
K at one slot: [num_local_kv_heads, head_dim]
V at one slot: [num_local_kv_heads, head_dim]
```

因此一层一个 slot 的普通 payload 是 $2N_{kv,local}De$ bytes；跨 $L_{local}$ 层是 $2L_{local}N_{kv,local}De$。在同一 cache group/config 中 slot 容量固定，换模型、dtype、TP、MLA/hybrid spec 或 backend 后则可能不同。

再核对边界：$B=16$、table `[5,2]` 时，position 15→block 5 offset 15→slot 95；position 16→block 2 offset 0→slot 32；position 23→block 2 offset 7→slot 39。logical position 连续不要求 flat slot 连续。

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

### 可执行的最小 allocator

下面的实现刻意小到可以逐行核对，但包含 `add_request`、`grow`、`free_request` 和 `physical_slot` 四条关键路径：

```python
from collections import deque
from math import ceil

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free = deque(range(num_blocks))
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    def _take(self, n: int) -> list[int]:
        # 原子检查：不足时一个 block 也不拿。
        if n > len(self.free):
            raise MemoryError(f"need {n}, free {len(self.free)}")
        return [self.free.popleft() for _ in range(n)]

    def add_request(self, request_id: str, num_tokens: int) -> None:
        if request_id in self.tables:
            raise KeyError(request_id)
        needed = ceil(num_tokens / self.block_size)
        blocks = self._take(needed)
        self.tables[request_id] = blocks
        self.lengths[request_id] = num_tokens

    def grow(self, request_id: str, new_tokens: int) -> None:
        old_len = self.lengths[request_id]
        new_len = old_len + new_tokens
        old_n = ceil(old_len / self.block_size)
        new_n = ceil(new_len / self.block_size)
        extra = self._take(new_n - old_n)
        self.tables[request_id].extend(extra)
        self.lengths[request_id] = new_len

    def free_request(self, request_id: str) -> None:
        for block_id in self.tables.pop(request_id):
            self.free.append(block_id)
        del self.lengths[request_id]

    def physical_slot(self, request_id: str, position: int) -> int:
        if not 0 <= position < self.lengths[request_id]:
            raise IndexError(position)
        logical = position // self.block_size
        offset = position % self.block_size
        physical = self.tables[request_id][logical]
        return physical * self.block_size + offset

a = BlockAllocator(num_blocks=6, block_size=4)
a.add_request("A", 6)
assert a.tables["A"] == [0, 1]
a.add_request("B", 3)
assert a.tables["B"] == [2]
a.free_request("B")
a.grow("A", 3)
assert a.tables["A"] == [0, 1, 3]
a.add_request("C", 5)
assert a.tables["C"] == [4, 5]
a.free_request("A")
a.add_request("D", 10)
assert a.tables["D"] == [2, 0, 1]
assert a.physical_slot("D", 9) == 1 * 4 + 1 == 5

before = (list(a.free), {k: v[:] for k, v in a.tables.items()})
try:
    a.grow("D", 100)
except MemoryError:
    pass
after = (list(a.free), {k: v[:] for k, v in a.tables.items()})
assert before == after                 # 失败不能留下半次分配
```

输入是 block 总数、block size 和请求长度；输出是请求到 physical block ids 的映射以及一维 slot。长期状态为 `free/tables/lengths`。position $p$ 的 shape 是标量，但其地址变换对应

$$
b_l=\lfloor p/B\rfloor,\quad o=p\bmod B,\quad
b_p=\mathrm{table}[b_l],\quad slot=b_pB+o.
$$

真实 vLLM 还包含 cache groups、prefix hash、引用计数、null block、抢占、lookahead slots、外部 KV connector 和并发安全；教学实现只能证明分页映射与原子容量检查，不能代表生产 allocator。

### 手算练习：为什么 table `[7,3,11]` 不是三段地址

令 $B=4$，逐个位置列出来：

| logical position | logical block | offset | physical block | flat slot |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 7 | 28 |
| 3 | 0 | 3 | 7 | 31 |
| 4 | 1 | 0 | 3 | 12 |
| 8 | 2 | 0 | 11 | 44 |
| 9 | 2 | 1 | 11 | 45 |

logical position 单调增加，flat slot 可以跳来跳去。kernel 不能用 `base + position` 猜地址，必须消费 block table/slot mapping。

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

### block 生命周期与引用计数

```text
普通空闲
  ref=0, hash=None
      │ allocate/acquire
      ▼
正在使用
  ref>0
      │ 完整块获得 cache identity
      ▼
正在使用且可缓存
  ref>0, hash=H
      │ 最后一个 owner free
      ▼
可命中的驱逐候选
  ref=0, hash=H, 位于 free queue
      ├─ cache hit/touch → 从 free queue O(1) 删除，ref++
      └─ eviction → 从 hash map 删除旧 H，清身份并覆盖
```

引用计数解决“多个请求共享同一前缀块时，谁可以释放”的问题。每次 acquire 增加 `ref_cnt`，请求结束或重算回滚时减少；只有降到 0 才能进入可分配集合。`ref_cnt=0` 表示没有活跃 owner，不表示 payload 一定为空。

侵入式双向 free queue 把 `prev_free_block/next_free_block` 指针放在 block 对象内。cache hit 可能 touch 队列中间的某个 block；有前驱、后继指针时可在 $O(1)$ 删除，而普通数组需要移动元素或搜索。代价是必须维护“在队列中/不在队列中”的不变量。

启用 APC 时，队首通常是优先重用/驱逐候选，队列顺序体现近似 LRU 与释放顺序；前缀链中较深/较浅块的具体优先规则和 touch 时机要以 v0.20.0 `BlockPool`/coordinator 调用为准，不能只看 queue 类就断言完整策略。`touch` 最显眼的用途确实是 APC 命中一个 `ref=0` 的 cached block，但凡需要为已有 block 增加 owner、把它从可驱逐集合转为活跃集合，也适用同一引用语义；不能把 `get_new_blocks` 当成共享已有块的替代。

null block 是不能像普通物理块那样释放/覆盖的特殊占位，用来统一某些缺失或 padding 路径。教学代码若把它放回 free queue，会破坏保留语义。

多引用的教学伪代码：

```python
# 教学伪代码
def acquire(block):
    if block.ref_cnt == 0:
        free_queue.remove(block)       # intrusive O(1)
    block.ref_cnt += 1

def release(block):
    assert block.ref_cnt > 0
    block.ref_cnt -= 1
    if block.ref_cnt == 0:
        free_queue.append(block)       # hash 可继续保留

def evict(block):
    assert block.ref_cnt == 0 and not block.is_null
    if block.block_hash is not None:
        cache_map.remove(block.block_hash, block)
    block.block_hash = None
    return block
```

输入/输出是控制面 block 对象；长期状态是 ref count、free queue 与 hash map。伪代码不触碰 K/V tensor，只说明 ownership 和身份如何同步。

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

把每一步展开成教学伪代码：

```python
# Scheduler 进程：决定，而不执行 attention
for req in selected_requests:
    new_blocks = kv_cache_manager.allocate_slots(
        req, num_new_tokens[req.id])
    scheduler_output.record(req.id, ids(new_blocks))

# worker：把增量计划合入长期的 worker-side state
gpu_model_runner._update_states(scheduler_output)
for req_id, new_ids in scheduler_output.new_block_ids.items():
    block_table.append_row_blocks(req_id, new_ids)

# commit 后才是设备可消费的表示
block_table.commit(num_reqs)
metadata = backend_builder.build(
    positions=device_positions,
    block_table=block_table.device_tensor,
    query_start_loc=query_start_loc,
)
model(input_ids, positions, metadata)
```

Scheduler 输入是 Request 进度、free blocks 和 token budget；输出只含本轮 counts/ids。`_update_states` 修改 worker-side cached request rows；`commit` 把相关 CPU 表区域复制/更新为 device tensor；backend builder 再结合 packed positions 形成 kernel metadata。真实代码对 new/cached/finished requests、PP、spec decode、CUDA Graph padding、encoder inputs 和多个 cache groups有更多分支。

### allocation block size 与 kernel block size

“block”必须带上下文：allocator 用它表示调度/共享单位，kernel/backend 用它表示 cache layout 和寻址单位，两者通常需要满足兼容约束，但不能仅凭同名变量假设永远相等。最终以 `KVCacheSpec`、生成的 cache config 和选中 backend 的 metadata/layout 契约为准。

多个 cache groups 时，一个请求可能同时维护多张 block table，例如 full attention 层一组、sliding-window 或其他 cache spec 一组。SchedulerOutput 的嵌套结构与 runner 的 table 更新都必须保留 group 维度；将其扁平为一个 list 只适用于单组教学例子。

`_build_attention_metadata` 在 runner 已根据 `SchedulerOutput` 更新请求状态、准备好本轮 packed inputs/positions 和 block tables 之后、真正调用模型 attention backend 之前执行。它要把通用 runner 状态翻译成选中 backend 的 metadata，常见输入概念包括：

| 数据 | 作用 |
|---|---|
| 本轮 query positions | 算 RoPE、causal 边界和 slot |
| query start locations/ranges | 从扁平 tokens 切回每个请求 |
| context lengths / computed counts | 确定每个 query 可见多少历史 |
| device block tables | 读取历史 KV 的逻辑→物理映射 |
| slot mapping | 新 K/V 写到哪些物理 slots |
| cache-group mapping | 选择对应层/cache spec 的 table |
| parallel parameters | PCP/DCP/TP 等如何分片位置与 heads |
| backend-specific flags | graph padding、window、dtype/layout 等 |

这些字段不是靠猜，而是从三处交叉确认：调用点实际传入参数、backend metadata builder 的类型/字段、具体 forward/kernel 的消费端。字段名可能随 backend/版本变化。

`parallel_config` 是引擎的并行拓扑配置，描述 TP/PP/DP 以及可能的 context parallel 等大小、rank/group 和通信方式。它会影响本地 heads/layers、worker 数、position/sequence 切分与 collective，因而 slot mapping/metadata 可能需要它；它不是“GPU 是否支持并行”的布尔开关。

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

应同时记录代价：

- 每个请求末块仍最多浪费 $B-1$ slots；
- block table、hash、ref count 与 free queue 占 CPU metadata；
- 每轮需要把增量映射同步到 worker/device；
- kernel 多一次逻辑到物理的间接寻址，且非连续块可能影响访问组织；
- block 太小会加长 table、增加管理频率，太大又增加尾部浪费并降低 prefix 命中粒度。

因此实验应固定模型、dtype、prompt/output 分布和调度配置，对比有效 KV tokens、allocated slots、free blocks、preemption、吞吐、TTFT/ITL 与 kernel time。只看 `torch.cuda.memory_allocated()` 看不到 pool 内部利用率。

## 8. 不要问“PagedAttention 还是 FlashAttention”

它们处在不同层次：

- PagedAttention：KV 的分页组织、共享和寻址；
- FlashAttention：exact attention 的 IO-aware 计算方法；
- Continuous Batching：本轮组合哪些请求；
- APC：哪些完整前缀 blocks 可跨请求复用；
- KV 量化：每个缓存元素使用多少 byte。

一个 backend 可以同时消费 paged KV layout，并使用 FlashAttention/FlashInfer 风格 kernel。

## 9. 四层证据

本地或网上的旧 `docs/design/paged_attention.md` 可能明确说明它描述原始 vLLM 论文，而非当前代码。阅读时区分：

1. 通用/论文思想：logical-to-physical blocks、按需分配、共享；
2. 历史 kernel 材料：早期 CUDA kernel 的线程、warp 和 layout；
3. 固定版本静态事实：v0.20.0 的 BlockPool、KVCacheManager、BlockTable、GPUModelRunner 与 backend contract；
4. 动态事实：启动日志、trace 和 profiler 证明特定配置实际选择、执行及耗时。

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
