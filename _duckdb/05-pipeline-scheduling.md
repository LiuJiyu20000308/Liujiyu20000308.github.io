---
title: "05 · Pipeline、Event 与并行调度"
order: 5
summary: 解释物理算子树如何按阻塞边界切成 Pipeline，Event 如何表达依赖，TaskScheduler 又如何把工作分配给线程。
keywords: [Pipeline, MetaPipeline, Event, Task, TaskScheduler]
description: DuckDB Pipeline 执行模型、事件依赖与任务调度源码笔记。
---

## 物理算子树不能直接回答并行问题

物理计划通常画成一棵自底向上的树，但它没有直接说明：两个分支能否同时执行、Hash Join 何时允许开始 probe、Aggregate 何时完成合并，以及每个线程一次领取多少工作。

DuckDB 因此从物理树构造 Pipeline。一个典型 Pipeline 可以抽象为：

```text
Source → Operator → Operator → ... → Sink
```

- **Source** 主动产生 `DataChunk`，如 Table Scan 或已完成物化结果的扫描；
- **Operator** 接收输入并产生输出，如 Filter、Projection 和 Join Probe；
- **Sink** 消费数据并积累状态，如 Hash Join Build、Hash Aggregate、Sort 或 Result Collector。

相邻的流式算子可以放在同一个 Pipeline 内，一个 Chunk 被 Source 取出后连续向前推动，不需要把完整中间结果写回某个公共队列。

## Pipeline Breaker 从哪里出现

只要下游必须等上游积累出全局状态，就会形成执行边界：

- Hash Join 的 build 端要先建好哈希表，probe 端才能查询；
- Hash Aggregate 要先消费分组数据，之后才能扫描最终结果；
- 全局 Order By 要先收集并排序输入，之后才能输出；
- Materialized CTE 或结果收集器需要保存中间数据。

“阻塞”不等于整个数据库只能串行运行。它只表示存在一个依赖边：依赖该状态的 Pipeline 必须等待，但其他无依赖的 Pipeline 或同一 Pipeline 内的多个 Task 仍然可以并行。

## 一个 Hash Join 如何被切分

```sql
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.amount > 100;
```

可以简化成两个阶段：

```text
Pipeline A: Scan customers → build JoinHashTable (Sink)
                                      │ finalize
                                      ▼
Pipeline B: Scan orders → Filter → Hash Join probe → Result Sink
```

Pipeline B 依赖 Pipeline A 的 build/finalize 完成。若 `customers` 扫描可以分区，多个 Task 会分别构建线程局部状态，再在 Combine/Finalize 阶段合并为可供 probe 的全局结构。

多表 Join 仍然由二叉 Join 节点组成。每个 Hash Join 有自己的 build 状态，整个执行过程因而是一张带依赖的 Pipeline 图，而不是“先完整执行左子树，再完整执行右子树”的固定递归。

## `MetaPipeline`、`Pipeline`、`Event`、`Task`

这四层对象分别解决不同粒度的问题：

| 对象 | 负责什么 |
| --- | --- |
| `MetaPipeline` | 组织共享同一 Sink 的 Pipeline，并处理 Join build 等跨组依赖 |
| `Pipeline` | 定义 Source、连续 Operators、Sink 与可并行性 |
| `Event` | 建立 build、execute、finish/finalize 等阶段的依赖图 |
| `Task` | 可以放入调度队列、由某个线程实际执行的工作单元 |

`Executor` 遍历物理计划并创建这些对象，等待依赖满足的 Event 才会调度后续工作。`TaskScheduler` 维护任务队列与工作线程；调用查询的前台线程也可以参与执行，并不必把全部工作都交给后台线程。

## `PipelineExecutor` 的内层循环

一个 Pipeline Task 最终由 `PipelineExecutor` 推动。它大致重复：

1. 从 Source 获取一个 Chunk；
2. 依次执行中间 Operator；
3. 将结果交给 Sink；
4. 根据返回状态决定继续取输入、继续产出剩余结果、暂时阻塞或结束；
5. Pipeline 完成时 flush 缓存算子并提交局部状态。

Operator 不一定“一批输入对应一批输出”。Join 的一个输入 Chunk 可能产生多批结果，某些算子也可能暂时没有输出。因此接口需要区分 `NEED_MORE_INPUT`、`HAVE_MORE_OUTPUT`、`FINISHED` 和 `BLOCKED` 等状态，执行器据此保存进度并恢复，而不是假设固定的一进一出。

## Global State 与 Local State

并行算子通常把状态分成两类：

- **Local state**：每个执行线程或 Task 私有，热路径无需频繁加锁；
- **Global state**：所有局部结果最终汇入，保存共享进度或最终结构。

Source、Operator、Sink 分别可以定义自己的状态。例如并行扫描的 GlobalSourceState 负责分配尚未扫描的数据区间，LocalSourceState 保存线程当前进度；Hash Aggregate 的 LocalSinkState 保存局部哈希表，Combine 再把它合并到全局状态。

这种设计的基本原则是：热循环尽量在线程本地完成，把同步集中到任务领取、局部合并和阶段切换处。否则线程越多，锁竞争和 Cache Line 抖动越严重。

## 并行度为什么不会无限增加

线程数只是上限，实际并行度还受这些因素限制：

- Source 能否切出足够多的独立工作；
- Pipeline 中的算子是否支持并行；
- 数据量是否足以摊薄 Task 创建与状态合并开销；
- 是否被单个 build/finalize 阶段形成关键路径；
- 内存带宽、缓存、NUMA 或临时文件 I/O 是否已饱和。

因此“CPU 没跑满”不能直接归结为线程数设置错误。应该先看 Pipeline 依赖图和每阶段可用 Task 数，再判断瓶颈是调度、同步、数据倾斜还是算子自身。

## 源码阅读入口

- `src/include/duckdb/parallel/meta_pipeline.hpp`：Pipeline 组与 Join build 依赖。
- `src/include/duckdb/parallel/pipeline.hpp`：Source/Operator/Sink 组成与任务调度。
- `src/include/duckdb/parallel/event.hpp`：事件完成、依赖与后继调度。
- `src/include/duckdb/parallel/pipeline_executor.hpp`：Chunk 推动与可恢复执行状态。
- `src/include/duckdb/parallel/task_scheduler.hpp`：任务队列和工作线程。
- `src/include/duckdb/execution/physical_operator_states.hpp`：Global/Local 状态接口。

## 面试时容易被追问

**Pipeline 和线程是什么关系？** Pipeline 是可流式执行的逻辑工作段，不等于固定线程。一个 Pipeline 可以产生多个 Task，被不同工作线程执行；同一线程也会先后执行多个 Pipeline 的 Task。

**Sink 为什么不一定意味着查询结束？** Sink 只表示当前 Pipeline 的数据消费端。Hash Join Build、Aggregate 和 Sort 都是 Sink，完成后其状态还会成为后续 Pipeline 的 Source 或 Operator 输入。

**Local/Global State 为什么比共享哈希表直接加锁更好？** 局部状态减少热路径同步和 Cache 争用；批量 Combine 的同步频率远低于逐行更新共享结构，但会付出额外局部内存和合并成本。

## 延伸阅读

- [DuckDB Parallel Grouped Aggregation](https://duckdb.org/2022/03/07/aggregate-hashtable.html)
- [DuckDB Execution Format](https://duckdb.org/docs/stable/internals/vector)
- [Morsel-Driven Parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf)
