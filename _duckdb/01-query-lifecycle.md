---
title: "01 · 整体架构与查询生命周期"
order: 1
summary: 一条 SQL 如何依次经过解析、绑定、优化、物理计划生成，并最终被拆成可以并行执行的任务。
keywords: [Parser, Binder, Logical Plan, Physical Plan, Executor]
description: DuckDB 整体架构与 SQL 查询生命周期源码笔记。
---

## 先建立一张全局地图

DuckDB 是嵌入进应用进程的分析型数据库。它没有必须单独部署的数据库服务，应用可以通过 C/C++ API、Python、命令行等入口直接创建连接并提交 SQL；但“嵌入式”只描述部署形态，内部仍然包含完整的 SQL 前端、优化器、执行引擎、事务和存储系统。

一条查询的主路径可以压缩成下面这条链：

```text
SQL text
  ↓
Parser → SQLStatement / ParsedExpression
  ↓
Binder → BoundStatement + LogicalOperator tree
  ↓
Optimizer → optimized LogicalOperator tree
  ↓
PhysicalPlanGenerator → PhysicalOperator tree
  ↓
Executor → MetaPipeline / Pipeline / Event / Task
  ↓
PipelineExecutor → DataChunk stream → QueryResult
```

这几个阶段解决的是不同问题，边界不能混在一起：

| 阶段 | 核心问题 | 典型输出 |
| --- | --- | --- |
| Parser | 这段文本在语法上表示什么 | 未绑定的语法树 |
| Binder | 表、列、函数和类型具体指向谁 | 带类型与列绑定的逻辑计划 |
| Optimizer | 在语义等价的前提下怎样减少工作量 | 重写后的逻辑计划 |
| Physical planner | 每个关系算子具体选择什么算法 | 物理算子树 |
| Executor | 怎样组织依赖、并行和数据流 | Pipeline/Event/Task 图 |
| PipelineExecutor | 一个数据批次如何流过各算子 | `DataChunk` 与最终结果 |

## 用一条查询贯穿各阶段

以这条分析查询为例：

```sql
SELECT c.region, SUM(o.amount) AS revenue
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.id
WHERE o.order_date >= DATE '2025-01-01'
GROUP BY c.region
ORDER BY revenue DESC
LIMIT 10;
```

Parser 只知道 `orders`、`customer_id` 和 `SUM` 是语法节点。Binder 查询 Catalog，确认表和列是否存在，解析别名作用域，确定 `amount` 的类型，并为 `SUM` 选择正确的函数重载。此后名字不再只是字符串，而会变成稳定的列绑定和类型信息。

逻辑计划表达“做什么”，大致包含 Scan、Filter、Join、Aggregate、Order 和 Limit。优化器可能把日期过滤下推到 `orders` 扫描，把不需要的列提前裁掉，根据统计信息调整 Join 顺序，并将 `ORDER BY + LIMIT` 改写为 Top-N。物理计划再为等值 Join 选择 Hash Join，为分组选择 Hash Aggregate。

执行阶段不会简单地从物理树根部递归调用到底。Hash Join 的 build 端必须先建立哈希表，Aggregate 和 Order 也需要积累状态，因此物理树会按阻塞边界切成多个 Pipeline，并通过 Event 建立先后依赖。每个可并行 Pipeline 再产生 Task，由调度器交给工作线程执行。

## 三棵树和一张执行图

阅读源码时最容易混淆的是“计划”和“执行”并非一个对象一路改到底：

1. **Parsed tree** 保留 SQL 语法结构，名称尚未绑定。
2. **Logical operator tree** 表达关系代数与已绑定表达式，是主要优化对象。
3. **Physical operator tree** 已决定 Hash Join、Hash Aggregate、Top-N 等执行算法。
4. **Pipeline dependency graph** 从物理树派生，表达实际并行单元和依赖关系。

物理算子树仍然适合解释查询结构；Pipeline 图才适合回答“哪些工作能同时进行、哪个阶段必须等待、一个 Task 具体执行什么”。

## `EXPLAIN` 应该看什么

`EXPLAIN` 的价值不是记住输出格式，而是验证自己的心智模型：过滤有没有下推、Join 的 build/probe 关系是否合理、基数估计和真实行数是否偏离、排序是否被 Top-N 替代。`EXPLAIN ANALYZE` 会实际执行查询，因此还能观察算子耗时和真实基数，但单个算子的时间不应被机械地相加——并行 Pipeline 的运行区间可能重叠。

性能分析时可以沿同一条链逆向排查：

```text
总延迟异常
  → 哪个物理算子或 Pipeline 最重
  → 输入/输出基数是否异常
  → 优化器为何生成这个计划
  → Binder 给出的类型、列和函数是否符合预期
```

## 源码阅读入口

- `src/parser/`：SQL 解析结果与表达式节点。
- `src/planner/planner.cpp`、`src/planner/binder.cpp`：绑定并生成逻辑计划。
- `src/optimizer/optimizer.cpp`：按阶段运行逻辑优化器。
- `src/execution/physical_plan/`：从逻辑算子生成物理算子。
- `src/execution/executor.cpp`：初始化物理计划并调度执行。
- `src/parallel/`：Pipeline、Event、Task 与线程调度。

## 面试时容易被追问

**为什么 Parser 之后不能直接执行？** 因为语法正确不代表对象存在，也没有确定列作用域、表达式类型和函数重载；这些是 Binder 的职责。

**物理算子树为什么还要转换成 Pipeline？** 算子树描述数据依赖，却没有直接表达并行粒度和阻塞边界。Pipeline 将连续的流式算子组合起来，Event 再表达 Pipeline 之间的依赖。

**嵌入式数据库为什么仍然需要事务与 Buffer Manager？** 是否独立部署和是否需要一致性、并发控制、持久化、缓存管理是两个维度。嵌入式只减少了进程间通信和部署成本。

## 延伸阅读

- [DuckDB Internals: Overview](https://duckdb.org/docs/stable/internals/overview)
- [DuckDB EXPLAIN Statement](https://duckdb.org/docs/stable/guides/meta/explain)
- [DuckDB Repository](https://github.com/duckdb/duckdb)
