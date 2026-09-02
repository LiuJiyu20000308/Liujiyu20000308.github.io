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

### 可运行的 hash chain 教学实现

Python 自带 `hash()` 会受进程随机种子影响，下面用 SHA-256 做可重复演示。真实 vLLM 的序列化字段和 hash 算法以 v0.20.0 源码为准。

```python
import hashlib
import json

def digest(payload) -> str:
    raw = json.dumps(payload, sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def block_hashes(token_ids, block_size, *, salt="", lora_id=None,
                 mm_hashes=(), cache_group=0):
    parent = "ROOT"
    out = []
    full = len(token_ids) // block_size
    for index in range(full):
        tokens = token_ids[index*block_size:(index+1)*block_size]
        identity = {
            "parent": parent,
            "tokens": tokens,
            "salt": salt,
            "lora_id": lora_id,
            "mm_hashes": list(mm_hashes),
            "cache_group": cache_group,
        }
        parent = digest(identity)
        out.append(parent)
    return out

a = block_hashes([1,2,3,4,5,6,7,8], 4)
b = block_hashes([9,9,9,9,5,6,7,8], 4)
assert a[1] != b[1]                 # 当前块相同，父历史不同
assert block_hashes([1,2,3,4,5], 4) == [a[0]]  # partial 不参与
assert block_hashes([1,2,3,4], 4, salt="tenant-B")[0] != a[0]
```

输入是 token ids、block size 和会改变缓存身份的额外字段；输出是每个完整块的链式 hash。长期状态没有变化。第二块身份是 $H_2=hash(H_1,tokens_2,extra)$。代码能证明父链、full-block 与 salt 的逻辑效果，不能证明真实 vLLM 使用相同 JSON 编码、SHA-256 或字段集合。

## 3. 为什么只缓存 full blocks

完整 block 的 token 内容不会再增长，hash 身份稳定。partial block 之后追加 token 会改变其内容：

```text
[I J _ _] → [I J K _] → [I J K L]
```

完整命中状态表如下：

| hash 已算出 | map 有对应 block | 身份兼容 | block 未被覆盖 | 结果 |
|---|---|---|---|---|
| 否 | — | — | — | 不能查找 |
| 是 | 否 | 是 | — | miss，需分配/计算 |
| 是 | 是 | 否 | 是 | miss，隔离正确 |
| 是 | 是 | 是 | 否 | 映射失效；这是实现 bug |
| 是 | 是 | 是 | 是 | hit，可 acquire/touch |

hash collision 在数学上并非不可能，生产实现的键结构/附加校验决定风险；教程不能把“hash 相同”提升为无需任何工程防护的绝对相等。

共同 prefix 长 $C$、block size 为 $B$ 时，纯 full-block 复用上界是

$$\left\lfloor C/B\right\rfloor B.$$

$C=50,B=16$ 时最多 48 tokens，尾部 2 tokens 重算。若一个请求在 partial block 未填满时永久结束，该尾块在这条普通 full-block APC 主线中不会获得稳定 hash，也不能作为独立 prefix-cache 单元被后续请求共享；它占用的 block 在请求释放后仍可作为普通物理空间重新分配。

“prompt 的所有完整 blocks 都在 cache”也不等于零次模型执行。KV 不保存“继续生成所需的最后位置 logits”；v0.20.0 的命中进度还受到 `num_tokens-1`、block alignment 和部分 speculative/EAGLE 分支限制。因此实现可能至少重算最后一个 token，某些对齐路径可能回退最后一整块。文章应说“跳过安全的已计算前缀”，而不是“全命中就不跑模型”。

partial 状态要区分三种计数：

```text
len(all_token_ids) % B       已知 token 的尾块
num_computed_tokens % B      KV 已真正写入的尾块
allocated slots              已预留但未必写过的空间
```

下一 KV 写入 logical position 是 `num_computed_tokens`；logical block/offset 为 `N//B,N%B`。Request 已能按已知 tokens 生成 hash，也不表示对应 KV 已 computed。

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

`extra_keys` 是“token 相同但 K/V 或共享策略仍可能不同”的通用附加身份，可包括 LoRA/adapter、multimodal feature/content hash、prompt embeddings 的内容摘要、cache group 等版本相关字段。`cache_salt` 是其中一种隔离 namespace：通常混入首块，差异再沿 parent hash 传播。salt 不是认证、加密或授权；服务仍需正确分配租户 salt 并控制访问。

真正未检测的 hash collision 理论上会导致错误 KV 复用，造成 silent wrong logits，跨租户时还可能成为安全问题。普通随机输入下宽 hash 碰撞概率很低，但面对不可信输入应核对 v0.20.0 的 hash 选项、字段和隔离策略；不要把 Python 内置 hash 当作密码学唯一证明。测试应断言“相同输入关系相等/不同父链不等”，不要写死受 `PYTHONHASHSEED` 影响的具体整数。

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

### longest-prefix lookup、touch、free 与 eviction

```python
# 教学伪代码：cache_map: hash -> physical block
def find_longest_prefix(block_hashes):
    hits = []
    for h in block_hashes:              # 必须从第一个 block 连续查
        block = cache_map.get(h)
        if block is None:
            break                       # 中间 miss 后不能跳到后面
        hits.append(block)
    return hits

def touch(block):
    if block.ref_cnt == 0:
        free_queue.remove(block)         # 它此前是驱逐候选
    block.ref_cnt += 1

def free(block):
    block.ref_cnt -= 1
    if block.ref_cnt == 0:
        free_queue.append(block)         # 保留 hash/KV 等待复用

def allocate_for_new_content():
    block = free_queue.popleft()
    assert block.ref_cnt == 0
    if block.block_hash is not None:
        del cache_map[block.block_hash]  # 覆盖前先删除旧身份
    block.block_hash = None
    return block
```

输入是新请求的完整块 hashes；输出是从 block 0 开始的最长连续 physical blocks。长期状态包括 map、free queue、hash 与 ref count。后一个块即使“自己 token 相同”也不能越过前面的 miss，因为它的 hidden/KV 依赖完整父历史。

### hash 产生、KV 写完与 cache 提交是三个时刻

```text
Request token 数达到 full-block 边界
  → update_block_hashes：知道“如果有效，它的身份是什么”
  → Scheduler 安排对应 tokens，worker 写 physical K/V
  → 执行/验证成功，computed progress 提交
  → cache_full_blocks：把 hash→physical block 加入内容索引
  → 后续请求才可安全命中
```

不是等整个 request 完成才提交；只要一个 full block 已安全计算完成并达到提交条件，活跃请求仍在继续时也可以被共享。反过来，speculative token 即使让 token list 凑满一块，在 target 接受前也不能暴露为稳定 cache。

### 一个 hash 为什么可能对应多个 physical blocks

`BlockHashToBlockMap` 的概念不是必须一对一。两个相同请求若并发 lookup 时都还没有看到“已完成且已提交”的 block，会各自 miss 并分配：

```text
t0: req A lookup H → miss → allocate physical 7
t0: req B lookup H → miss → allocate physical 12
t1: A/B 都完成 → H → {7,12}
```

这与 `block 7.ref_cnt=2` 不同：前者是两份相同 KV 副本，各有 owner；后者是一份 physical KV 被两个请求共享。后来的请求命中 H 时可选择某个可用副本并增加引用。

需要新 physical block 的典型条件是：prefix miss、当前可写块已满、请求需要自己的 partial 尾块、相同内容并发时尚无已提交副本、或旧 cache 已被驱逐。新 block 仍来自同一 `BlockPool`；free queue 只是 pool 中 `ref=0` blocks 的可分配索引和 LRU 顺序，不是第二个物理池。

### `[7,19,23]` 时间线中的数据与 shape

假设每 block 4 tokens，请求 `[A..H,I,J]`：

| 阶段 | computed tokens | block table | 本轮 GPU 输入 | 写入 |
|---|---:|---|---|---|
| lookup 前 | 0 | `[]` | — | — |
| 命中 H1/H2 | 8 | `[7,19]` | — | 不重写命中 KV |
| 分配尾块 | 8 | `[7,19,23]` | `[I,J]`，shape `[2]` | block 23 offsets 0,1 |
| prefill 后 | 10 | `[7,19,23]` | — | 尾块仍 partial |
| 采样新 token 后 | 10，逻辑 11 | 同表 | 下一轮才处理采样 token | 下一轮 offset 2 |

worker 不是从 token 文本“知道”只算 `[I,J]`，而是从 Scheduler 传来的 computed/scheduled counts、block ids 和 positions 构造 packed input。命中 blocks 提供 attention history；未命中后缀仍必须经过所有模型层。

## 6. `ref_cnt=0` 为什么仍可能是缓存

请求所有权与缓存内容有效性是两个维度：

| 状态 | 含义 |
|---|---|
| `ref_cnt>0` | 至少一个活跃请求正在使用，不可驱逐 |
| `ref_cnt=0` 且有 hash | 无活跃 owner，但仍可快速命中；容量紧张时可驱逐 |
| `ref_cnt=0` 且无 hash | 普通空闲 block |

这种设计让 APC 无需把所有历史内容永久锁在显存中：热前缀可以再次被 touch，冷前缀则为新请求让路。

APC 与其他缓存必须按“缓存什么”区分：

| 机制 | key | value | 命中后省掉什么 |
|---|---|---|---|
| 普通请求内 KV | 同一 request 的历史位置 | 每层 K/V | decode 的历史重算 |
| APC | 精确 token prefix + extra identity | 完整前缀 blocks 的 K/V | 命中部分 prefill |
| 语义缓存 | embedding/相似度与策略 | 常为最终答案 | 整段模型调用，可能近似 |
| 结果缓存 | 完整请求 key | 最终输出 | 相同请求的生成 |
| KV 量化 | 无“命中”概念 | 更低精度 K/V | 主要省容量/带宽 |

### 多 GPU 上 block pool 的“共享”含义

同一 TP/PP 模型副本通常由一套逻辑分配决策协调相同 block ids，但各 GPU 不共享同一块显存：

```text
logical block 7
  ├─ TP rank 0 local slot 7：本地 KV heads
  ├─ TP rank 1 local slot 7：另一组 KV heads
  └─ PP stage 1 local slot 7：该 stage layers 的 KV
```

`full/free/cached/evicted` 是逻辑 block 级状态。某个 rank 可以稍早结束 kernel，但不能单独把 local slot 7 释放并改存别的请求；所有必要 workers/stages 完成后，控制面才统一推进。否则同一 block id 在各 ranks 指向不同序列，会成为严重正确性错误并应触发 engine failure，而不是正常调度状态。

DP replicas 通常各有独立 Scheduler/KV pool；replica 0 的 block 7 与 replica 1 的 block 7 没有共享身份。跨 replica 复用需要额外 KV transfer/connector 机制，普通 APC 不自动做到。

cache group 用于把具有兼容 cache spec、block size/layout/attention 行为的一组层归在同一管理维度。hybrid 模型可能有 full-attention、sliding-window 或其他状态类型，不能用一张 table/一个 hash 身份混管。hash 与 block id 需要带 group identity，Scheduler/runner 也要同步每组映射。

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

counter 若是进程生命周期累计值，必须做前后快照差分：

```python
def delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after}

before = {"queries": 1727, "hits": 0}
after  = {"queries": 3456, "hits": 1712}
d = delta(before, after)
rate = d["hits"] / d["queries"]
assert d == {"queries": 1729, "hits": 1712}
assert abs(rate - 0.9901677) < 1e-6
print(f"{rate:.2%}")                # 99.02%
```

输入是同一 counter 在请求前后的累计快照；输出才是该请求/区间的增量命中率。它不能告诉我们每个命中 block 节省了多少 wall time。

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

客户端实验伪代码应把计时边界和 cache 状态写明：

```python
# 教学伪代码
restart_server(enable_prefix_caching=True)   # 各大组清状态
wait_until_ready()
warm_up(unrelated_prompt)

for prefix_len in [128, 512, 2048]:
    for output_len in [1, 32, 256]:
        first = request(prefix(prefix_len) + question_a,
                        max_tokens=output_len, timestamps=True)
        counters_before = scrape_metrics()
        second = request(prefix(prefix_len) + question_b,
                         max_tokens=output_len, timestamps=True)
        counters_after = scrape_metrics()
        record(ttft=second.first_token-second.start,
               itl=inter_token_intervals(second),
               e2e=second.end-second.start,
               counter_delta=delta(counters_before, counters_after))
```

APC on/off 不能只在同一已污染进程中切一个布尔值；要控制 pool 历史、请求顺序、warmup、并发和采样。短 output 更容易让 prefill 节省显现在 E2E；长 output 中 decode 可能主导。

开头第一个 token 不同会让 $H_1$ 不同，链式 $H_2,H_3,\ldots$ 也全不同，所以后面相同的 blocks 不能命中。这是 causal state 依赖，不是 hash 算法“不够聪明”。

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
