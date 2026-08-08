---
title: "03 · 优化器与计划重写"
order: 3
summary: 理解规则优化、统计信息与 Join Order 如何协作，以及如何判断一个优化究竟减少了什么工作量。
keywords: [Expression Rewrite, Filter Pushdown, Join Order, Cardinality, TopN]
description: DuckDB 查询优化器、基数估计与 Join Order 源码笔记。
---

## 优化器的目标不是“让树更漂亮”

优化器接收已经绑定的逻辑计划，在结果语义不变的前提下减少扫描数据、表达式计算、中间结果、内存占用或阻塞时间。判断一次改写是否有价值，关键不是节点数量，而是它让执行引擎少做了什么。

DuckDB 的优化过程同时包含：

- **规则驱动优化**：常量折叠、表达式化简、Filter/Limit 下推、无用列删除、公共表达式处理等；
- **代价相关优化**：根据统计信息估计基数，选择 Join 顺序和 build/probe 侧；
- **物理友好改写**：例如将 `ORDER BY + LIMIT` 变为 Top-N，避免完整排序全部结果。

不同阶段不是可以随意交换的。比如先做 Filter Pushdown，后续 Join Order 才能看到更小的输入基数；先删除无用列，物理算子才可能减少向量宽度与内存流量。

## 四类最重要的优化

### 1. 表达式重写

表达式重写通常不改变逻辑树结构，只化简节点内部的表达式。例如常量折叠、布尔表达式化简、可安全执行的代数变换。收益是减少每个 `DataChunk` 上重复执行的工作。

```sql
WHERE (price > 100 AND TRUE) AND 2 + 3 = 5
```

常量部分可以在规划期求值，执行期只保留真正依赖列值的判断。这里必须谨慎处理 NULL、溢出、错误抛出时机和非确定性函数，不能直接套用普通代数恒等式。

### 2. Filter Pushdown 与列裁剪

```sql
SELECT customer_id
FROM orders
WHERE status = 'PAID';
```

过滤越早执行，后续算子处理的行越少；只读取 `customer_id` 和 `status`，数据批次也更窄。若数据源支持 Filter/Projection Pushdown，扫描端甚至可以少读取存储块或外部文件列。

但过滤不能无条件穿过所有算子。Outer Join、窗口函数、聚合和具有副作用的表达式都可能改变可下推边界。例如将针对右表的谓词错误推到 Left Join 下方，可能把本应保留的 NULL 扩展行删除。

### 3. Join Order 与 build/probe 选择

三个表的 Inner Join 在逻辑上可以有多种等价顺序，但中间结果大小可能相差几个数量级。优化器利用表基数、列统计、过滤选择率和 Join 图估计候选计划代价，尽量先执行高选择性的过滤与连接。

对 Hash Join，较小一侧通常更适合作为 build 端，因为哈希表的大小直接影响内存、缓存命中和是否需要分区/落盘。不过“原表更小”不等于“过滤后更小”，build/probe 选择应基于进入该算子的估计基数和行宽。

### 4. Top-N 与物化时机

```sql
SELECT * FROM events ORDER BY score DESC LIMIT 100;
```

完整排序需要保存并排序全部候选行；Top-N 只维护当前最优的有限集合。当 N 远小于总行数时，时间和内存都明显下降。类似地，列的物化越晚，前面算子搬运的数据越少，但过度延迟也可能引入额外索引或重取成本。

## 基数估计为什么决定上限

优化器看不到未来的真实行数，只能根据统计信息估计。估计会被以下情况破坏：

- 多列高度相关，却被近似为独立；
- 数据倾斜或长尾分布被简单统计量掩盖；
- 谓词包含复杂函数，无法推导选择率；
- Join Key 分布不均或存在大量重复值；
- 外部数据源缺少统计信息。

估计错误会连锁放大：Join 顺序错误 → 中间结果变大 → Hash Table/排序内存上升 → 缓存局部性下降或触发落盘。分析慢查询时，比较 estimated cardinality 与 actual cardinality 往往比只看“哪个算子最慢”更接近根因。

## 评估一个优化的正确方式

一个可复现的优化实验至少需要：

1. 保存优化前后的逻辑/物理计划；
2. 对比每个关键算子的输入、输出与估计基数；
3. 固定数据、线程数、内存限制和冷热缓存条件；
4. 多次运行并报告分布，而非只挑最快一次；
5. 同时观察延迟、CPU、峰值内存和临时文件，而非只看 wall time；
6. 用结果校验或差分测试证明语义没有变化。

这也是数据库优化与普通代码微优化的区别：一次计划重写可能让算法复杂度发生变化，收益大；一旦 NULL、重复行或 Outer Join 语义处理错误，影响也更严重。

## 源码阅读入口

- `src/optimizer/optimizer.cpp`：内置优化阶段的组织顺序。
- `src/optimizer/filter_pushdown/`：谓词下推与边界判断。
- `src/optimizer/join_order/`：Join 图、基数估计与顺序搜索。
- `src/optimizer/statistics_operator/`：统计信息传播。
- `src/optimizer/remove_unused_columns.cpp`：列裁剪。
- `src/optimizer/topn_optimizer.cpp`：Order + Limit 的 Top-N 改写。

## 面试时容易被追问

**规则优化器和基于代价的优化器有什么区别？** 前者匹配局部结构并应用通常有利的等价规则；后者需要枚举候选方案并借助统计信息比较代价。实际系统会组合使用。

**Filter 越早越好吗？** 大多数情况下是，但必须满足语义等价；此外极低成本、低选择性的过滤也可能不值得破坏向量化或数据源的顺序特性。

**为什么优化后的单个算子可能更慢，总查询却更快？** 改写可能把工作集中到更少的算子，或让某个算子承担额外过滤，却显著减少后续数据量。评估应看端到端代价和关键资源，而不是孤立的算子时间。

## 延伸阅读

- [DuckDB Optimizers: The Low-Hanging Fruit of Query Optimization](https://duckdb.org/2024/11/14/optimizers.html)
- [DuckDB EXPLAIN ANALYZE](https://duckdb.org/docs/stable/guides/meta/explain_analyze)
- [DuckDB Join Operations](https://duckdb.org/docs/stable/guides/performance/join_operations)
