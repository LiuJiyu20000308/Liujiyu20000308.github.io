---
title: "06 · Join、Aggregation 与阻塞算子"
order: 6
summary: 以 Hash Join 和分组聚合为主线，理解 build/probe、局部状态合并、数据倾斜、NULL 语义与算法选择。
keywords: [Hash Join, Build Probe, Hash Aggregate, Radix Partitioning, Skew]
description: DuckDB Hash Join、聚合算子与数据倾斜源码笔记。
---

## Hash Join：先 build，再 probe

对等值连接：

```sql
SELECT o.order_id, c.region
FROM orders o
JOIN customers c ON o.customer_id = c.id;
```

Hash Join 通常分为两个阶段：

1. **Build**：读取一侧连接键和所需 payload，计算哈希并建立 `JoinHashTable`；
2. **Probe**：读取另一侧的 Chunk，批量计算键哈希，在哈希表中查找候选并校验连接条件。

较小的一侧通常被选作 build 端，因为哈希表需要驻留内存且会被反复随机访问。这里的“小”不仅是行数，还包括实际 payload 宽度、过滤后的基数、键分布与额外状态。优化器的 build/probe 调整如果估计失误，执行阶段就可能出现明显内存压力。

DuckDB 的 Join Hash Table 使用开放寻址/线性探测相关结构组织哈希槽，并保存匹配所需的 tuple data。具体布局会随版本演进，阅读时应把不变量抓住：先用哈希缩小候选，再检查真正的 Join 条件；键冲突和重复键都不能只靠哈希值判断。

## 一个输入 Chunk 为什么可能产生多批输出

若一个 probe key 对应 build 侧大量重复键，一行输入可能匹配成千上万行。输出 Chunk 容量有限，Join 必须保存扫描位置，先返回一批结果，下次调用继续产出。这就是 `HAVE_MORE_OUTPUT` 状态存在的重要原因。

数据倾斜会产生三类影响：

- 某些 hash bucket 或分区异常大，Cache 局部性变差；
- 单个 probe Chunk 的输出爆炸，后续算子工作量大幅增加；
- 并行分区负载不均，少数 Task 成为尾部延迟。

因此诊断 Join 不能只看两侧输入行数，还要看键的 distinct count、重复分布和输出基数。

## Join 类型决定状态与收尾逻辑

Inner Join 只输出匹配项；Left/Right/Full Outer Join 还要追踪未匹配行；Semi/Anti Join 只关心是否存在匹配；Mark Join 需要保留三值逻辑信息。它们都可能共享哈希查找框架，但输出语义和附加状态不同。

NULL 尤其容易出错：普通 `=` 中 NULL 不与 NULL 相等，`IS NOT DISTINCT FROM` 则有不同语义；Anti/Mark Join 还会受子查询中的 NULL 影响。优化或实现 Join 时必须从 SQL 语义推导，不能把空值简单当作一个普通哈希键。

## 不同条件为什么需要不同 Join 算法

Hash Join 适合可哈希的等值条件，但不是所有连接都应强行哈希：

- 小输入或一般非等值条件可能使用 Nested Loop/Blockwise Nested Loop；
- 单个有序比较条件可以使用 Piecewise Merge Join；
- 两个不等式条件可由 IEJoin 类算法处理；
- 时间序列最近匹配可使用 ASOF Join；
- Cross Product 没有连接谓词，只生成笛卡尔积。

物理计划选择的是与条件结构匹配的算法。性能比较要包含排序、build、物化等准备成本，不能只比较核心匹配循环。

## Hash Aggregation：把状态按 group key 归并

```sql
SELECT region, COUNT(*), SUM(revenue)
FROM sales
GROUP BY region;
```

Hash Aggregate 对每个输入 Chunk 批量计算 group key 的哈希，定位或创建 group，再调用各聚合函数的 update。每个 group 保存的不是所有原始行，而是聚合状态，例如 Count 的计数器、Sum 的累加状态、Average 的 sum 与 count。

并行执行通常经过：

```text
每线程 Local Hash Table
  → Combine / radix partition
  → Global grouped state
  → Finalize aggregate states
  → Scan result groups
```

DuckDB 的分组聚合使用 radix partitioning 将哈希空间切分，先让线程在局部状态上工作，再按分区合并。这样可以控制单次合并的工作集并减少全局锁竞争。若 group key 的取值范围很小且稠密，Perfect Hash Aggregate 还可以直接把键映射到数组位置，省去通用哈希表的探测成本。

## 聚合函数接口中的四个阶段

理解一个可并行聚合函数，至少要区分：

- **Initialize**：创建空状态；
- **Update**：用输入向量更新状态；
- **Combine**：合并两个部分状态；
- **Finalize**：从内部状态产生 SQL 结果。

Combine 决定了聚合能否高效并行。Count、Sum、Min/Max 很容易合并；精确分位数、复杂去重或用户自定义状态可能更大，合并成本也更高。数值聚合还要考虑浮点结合律并不严格成立：并行归并顺序变化可能带来末位差异，需要在性能、精度和确定性之间明确取舍。

## Sort 与 Window：另一类全局状态

Order By 通常需要收集排序键和 payload，生成局部 runs 后再归并；窗口函数常需按 partition/order 键重新组织数据。它们同样是 Pipeline Breaker，但与 Hash Aggregate 的状态结构不同。

分析这类算子时可以统一问四个问题：输入能否分区、局部状态是什么、何时合并/Finalize、最终结果如何重新成为 Source。掌握这套框架后，新算子的执行流程会更容易读懂。

## 源码阅读入口

- `src/execution/operator/join/physical_hash_join.cpp`：Build、Finalize、Probe 与外部 Hash Join。
- `src/include/duckdb/execution/join_hashtable.hpp`：JoinHashTable 与 probe 扫描状态。
- `src/execution/operator/join/`：各类物理 Join 实现。
- `src/execution/operator/aggregate/physical_hash_aggregate.cpp`：分组聚合 Sink/Source 生命周期。
- `src/include/duckdb/execution/radix_partitioned_hashtable.hpp`：Radix 分区聚合状态。
- `src/execution/operator/order/`：Sort 与 Top-N 物理算子。

## 面试时容易被追问

**Hash Join 为什么通常让小表 build？** Build 哈希表占用内存并位于 probe 热路径，小表通常降低内存与随机访问工作集；但真正依据应是过滤后的基数、行宽和键分布。

**三个表连接会共用一张哈希表吗？** 通常不会。物理计划仍是二叉 Join 树，每个 Hash Join 节点维护自己的 build 状态和依赖；上一个 Join 的输出可以成为下一个 Join 的输入。

**Hash Aggregate 为什么使用线程局部表？** 避免每行更新共享表时加锁；代价是额外内存和 Combine，因此局部表大小、分区方式与合并策略都影响扩展性。

## 延伸阅读

- [DuckDB Parallel Grouped Aggregation](https://duckdb.org/2022/03/07/aggregate-hashtable.html)
- [DuckDB Join Operations](https://duckdb.org/docs/stable/guides/performance/join_operations)
- [DuckDB ASOF Join](https://duckdb.org/docs/stable/guides/sql_features/asof_join)
