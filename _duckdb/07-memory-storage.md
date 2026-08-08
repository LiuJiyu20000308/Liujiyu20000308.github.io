---
title: "07 · 内存、存储与落盘"
order: 7
summary: 区分 Buffer Manager、算子临时状态与进程内存，理解列式存储、MVCC、WAL，以及大于内存查询如何通过临时文件继续执行。
keywords: [Buffer Manager, Spilling, Columnar Storage, MVCC, WAL]
description: DuckDB Buffer Manager、列式存储、事务和大于内存查询笔记。
---

## 先区分三类内存

分析 DuckDB 内存问题时，不能把进程 RSS、`memory_limit` 和某个算子的峰值状态当成同一个概念：

1. **Buffer-managed memory**：通过 Buffer Manager 管理的数据页和可驱逐临时块，受配置的内存限制约束；
2. **Operator state**：Hash Table、排序 runs、聚合状态、结果 Chunk 等执行期对象，其中一部分会接入 Buffer Manager，一部分可能由普通分配器持有；
3. **进程总内存**：还包括线程栈、Allocator 缓存、扩展库、查询结果和应用自身内存。

因此 `memory_limit = 8GB` 不等于操作系统看到的进程 RSS 绝不会超过 8GB。排查 OOM 时要确认是哪一类对象增长，以及它是否支持驱逐或落盘。

## Buffer Manager 的 Pin/Unpin 模型

Buffer Manager 负责给数据库内部提供受控内存块，并在内存压力下选择可驱逐对象。核心关系可以简化为：

```text
BlockHandle  → 记录块的身份、状态和所有权
BufferHandle → 一次已 pin 的访问权，保证使用期间数据驻留内存
BufferPool   → 维护总配额与可驱逐队列
Temp storage → 保存不能直接丢弃的临时块
```

算子使用数据前 Pin，持有 `BufferHandle` 期间块不能被驱逐；访问结束后 Unpin，块才重新成为候选。可从原始数据重新计算的临时缓冲可以直接销毁，不可丢失的中间状态则需要写入临时文件，之后再 Pin 回内存。

设计执行算子时，长时间 Pin 大量块会让 Buffer Manager 失去腾挪空间；绕过管理器分配巨型状态，又会使内存限制和落盘机制失效。

## 大于内存查询如何继续运行

支持 external execution 的算子会在内存不足时把部分状态分区并写到 `temp_directory`，之后逐分区读取处理。典型场景包括大型排序、分组聚合、Hash Join 和窗口计算。

以外部 Hash Join 为例，核心思想不是把完整哈希表随意换出，而是按哈希高位将 build/probe 数据切成对应分区：每次只把一个能放入内存的 build 分区建表，再处理对应 probe 分区。额外 I/O 换来了受控峰值内存。

外部算法的实际性能取决于：

- 分区是否均匀，是否存在一个无法缩小的倾斜大分区；
- 临时目录所在磁盘的带宽、延迟和剩余空间；
- 每轮分区/序列化产生的 CPU 与内存拷贝；
- 计划是否能更早过滤和裁剪列，直接避免落盘。

可以显式设置资源边界：

```sql
SET memory_limit = '8GB';
SET temp_directory = '/fast-ssd/duckdb-tmp';
SET max_temp_directory_size = '100GB';
SET threads = 4;
```

降低线程数有时能缓解内存压力，因为并行 Task 的局部状态会同时存在；它不是通用提速方法，而是一种用并行度换峰值内存的手段。

## 列式存储与 Row Group

DuckDB 持久化数据按列组织，表再被划分为 Row Group，列内部由 Column Segment 组成并可采用不同压缩方式。分析查询只读取所需列；Zone Map 等统计信息还可以跳过不可能满足谓词的段。

Row Group 是存储与并行扫描的重要粒度：足够大的表可以把不同 Row Group 分给多个扫描 Task。数据布局因此同时影响 I/O、压缩率、并行度和更新成本。

向量化执行的 `DataChunk` 与持久化 Column Segment 不是同一个对象。前者是执行期批次，后者是存储布局；扫描算子负责把存储段解码/引用成 Vector。把这两个层次分清，才能判断优化是在减少磁盘读取、解压工作，还是只减少执行阶段的内存搬运。

## 事务、MVCC 与 WAL

DuckDB 使用 MVCC 让读事务看到一致快照，同时允许事务提交新的版本。更新或删除不能直接让正在读取旧快照的查询失去数据；系统需要保存版本可见性信息，并在安全时回收旧版本。

对持久化数据库，WAL 先记录尚未 checkpoint 到主数据库文件的变更，用于崩溃恢复；Checkpoint 再把稳定状态整理回数据库文件并缩短恢复路径。MVCC 解决并发可见性，WAL/Checkpoint 解决持久性和恢复，它们不是同一个机制。

DuckDB 的并发模型首先面向单进程内的并发连接。多个进程同时写同一个数据库文件不是它的主要设计目标；跨进程使用时必须遵守官方并发约束，而不能照搬客户端/服务器数据库的连接方式。

## OOM 与落盘性能的排查清单

1. 物理计划中哪个算子保存全局状态，输入基数和行宽是多少；
2. estimated/actual cardinality 是否严重偏离；
3. Join build 侧、group key 或 order payload 是否带了无用列；
4. 键分布是否倾斜，单一分区是否异常大；
5. 当前线程数产生了多少份 Local State；
6. 临时目录是否启用，容量、带宽和文件系统是否正常；
7. 进程增长来自 Buffer Manager 还是外部库/应用结果对象；
8. 是否能通过早过滤、列裁剪、分批执行或预聚合改变问题规模。

## 源码阅读入口

- `src/include/duckdb/storage/buffer_manager.hpp`：Buffer Manager 抽象接口。
- `src/include/duckdb/storage/standard_buffer_manager.hpp`：内存块、驱逐与临时文件实现。
- `src/include/duckdb/storage/buffer/`：BlockHandle、BufferHandle 和 BufferPool。
- `src/storage/table/`：Row Group、Column Segment 与表存储。
- `src/transaction/`：事务、版本信息和提交/回滚。
- `src/storage/wal_replay.cpp`、`src/storage/checkpoint/`：恢复与 Checkpoint。

## 面试时容易被追问

**Buffer Manager 和操作系统 Page Cache 有什么区别？** 操作系统管理文件页，却不了解查询算子状态、块是否可重算或何时能安全驱逐；数据库 Buffer Manager 用语义信息做配额、Pin/Unpin 和临时数据管理，两者会共同影响实际内存。

**为什么落盘后不一定只是“慢一点”？** 若数据倾斜导致最大分区仍放不进内存，可能反复 repartition；临时盘不足或随机 I/O 很慢也会让延迟非线性恶化。

**MVCC 和 WAL 分别解决什么？** MVCC 决定并发事务能看到哪个版本；WAL 记录未 checkpoint 的持久化变更，支持崩溃恢复。

## 延伸阅读

- [DuckDB Tuning Workloads](https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads)
- [DuckDB Out of Memory Errors](https://duckdb.org/docs/stable/guides/troubleshooting/oom_errors)
- [DuckDB Storage Internals](https://duckdb.org/docs/stable/internals/storage)
- [DuckDB Concurrency](https://duckdb.org/docs/stable/connect/concurrency)
