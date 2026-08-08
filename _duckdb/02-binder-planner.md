---
title: "02 · Binder 与逻辑计划"
order: 2
summary: 从只有名字的语法树出发，理解作用域解析、类型推导、函数重载与 ColumnBinding 如何构成后续优化的语义基础。
keywords: [Binder, Catalog, Type Resolution, ColumnBinding, LogicalOperator]
description: DuckDB Binder、类型绑定与逻辑计划源码笔记。
---

## Binder 真正解决什么

Parser 能判断 `SELECT a + 1 FROM t` 符合 SQL 语法，却不知道 `t` 属于哪个 Catalog/Schema、`a` 是否存在、`a + 1` 应返回什么类型。Binder 的任务是把“按名字描述的查询”转换成“指向真实对象、类型完整的查询”。它是 SQL 语义和执行计划之间的边界。

绑定主要包含五类工作：

1. 在 Catalog 中解析表、视图、函数、类型等对象。
2. 建立子查询、CTE、表别名和列别名的作用域。
3. 将列名转换为内部列绑定，并检查歧义与可见性。
4. 推导表达式类型，插入必要的隐式 Cast，选择函数重载。
5. 收集相关列和参数信息，为子查询改写与 Prepared Statement 提供依据。

这里越严格，后续优化器越简单。优化阶段可以相信输入已经通过类型检查，不必每次重写表达式都重新猜测一个名字的含义。

## 名字如何变成 `ColumnBinding`

考虑：

```sql
SELECT a.id
FROM accounts AS a
JOIN events AS e ON a.id = e.account_id;
```

如果计划一直携带字符串 `a.id`，投影裁剪、Join 重排或子查询改写都可能让名字失去稳定含义。DuckDB 使用类似 `(table_index, column_index)` 的 `ColumnBinding` 标识逻辑计划中的列来源：前者区分关系实例，后者区分该关系输出中的列。

这不是存储层的物理列地址，而是逻辑计划中的身份标识。每个产生新输出的算子都要维护自己的 bindings；优化器改变计划结构后，还必须保证表达式引用与新输出一致。许多“计划看起来正确但执行结果错”的问题，本质上都是改写算子后没有正确维护列绑定。

## 作用域、别名与相关子查询

Binder 通过绑定上下文管理当前查询块可见的表和列。遇到 `id` 时，它要回答：

- 当前作用域是否只有一个 `id`；
- `a.id` 中的 `a` 是否为可见别名；
- 内层查询没有找到时，是否应向外层查询块查找；
- 这个引用是否因此成为 correlated column。

例如：

```sql
SELECT c.id
FROM customers AS c
WHERE EXISTS (
  SELECT 1
  FROM orders AS o
  WHERE o.customer_id = c.id
);
```

内层的 `c.id` 依赖外层行。Binder 必须记录引用深度与列信息，之后才能把相关子查询转换为可以执行的 Join/Dependent Join 结构。相关子查询的难点不在语法，而在于外层值怎样进入内层计划，以及改写后怎样保持 SQL 的 NULL 与重复值语义。

## 类型推导与函数重载

SQL 允许字面量、参数、不同整数宽度、Decimal、日期时间等类型共同参与表达式。Binder 会自底向上绑定表达式：先确定子表达式的候选类型，再根据运算符或函数签名决定公共类型并插入 Cast。

```sql
SELECT amount * 1.05, date_trunc('month', created_at)
FROM orders;
```

这里至少涉及数值公共类型、Decimal 精度规则、字符串字面量的目标类型，以及 `date_trunc` 的重载选择。类型选择不仅影响正确性，也会改变后续物理表示和算子代价：无意中的高成本 Cast 可能阻止过滤下推或增加每个数据批次的计算量。

Prepared Statement 中尚未赋值的参数更复杂。系统需要保留参数类型信息，在参数可确定后完成绑定或重新规划，不能把未知参数随意当成字符串。

## 从 Bound Node 到逻辑算子

完成表达式绑定后，Binder/Planner 会生成 `LogicalOperator` 树。常见节点包括：

- `LogicalGet`：表或表函数扫描；
- `LogicalFilter`：谓词；
- `LogicalProjection`：表达式计算与输出列；
- `LogicalComparisonJoin`：等值或比较 Join；
- `LogicalAggregate`：分组键与聚合表达式；
- `LogicalOrder`、`LogicalLimit`：排序与限制。

逻辑节点表达关系语义，不承诺具体算法。`LogicalComparisonJoin` 说明连接条件，却不等于最终一定使用 Hash Join；物理计划阶段还会根据条件类型和统计信息选择实现。

## 一个实用的 Binder 排错顺序

遇到 “column not found”、ambiguous reference 或函数签名不匹配时，可以按下面顺序检查：

1. 当前查询块有哪些绑定，别名是否遮蔽了原表名；
2. `SELECT *` 或 `table.*` 展开后实际有哪些列；
3. 表达式每一层的返回类型和隐式 Cast；
4. 函数候选重载与参数类型是否匹配；
5. 子查询中的列来自当前层还是外层。

## 源码阅读入口

- `src/include/duckdb/planner/binder.hpp`：Binder 状态、绑定上下文与相关列信息。
- `src/planner/binder/`：查询节点、表引用、表达式和语句的具体绑定。
- `src/include/duckdb/planner/bind_context.hpp`：名称与作用域解析。
- `src/planner/expression_binder/`：不同 SQL 子句中的表达式绑定规则。
- `src/include/duckdb/planner/logical_operator.hpp`：逻辑算子公共接口。

## 面试时容易被追问

**Binder 与 Optimizer 的边界是什么？** Binder 决定语义：对象是谁、类型是什么、列指向哪里；Optimizer 在语义已确定的前提下改变等价执行方式。

**为什么列不能只用名字标识？** 同名列、Self Join、子查询和投影改写都会让字符串产生歧义；内部 binding 为关系实例和输出位置提供稳定身份。

**隐式 Cast 为什么可能影响性能？** Cast 本身需要逐批计算，还可能让谓词不再能直接作用于存储列，削弱过滤下推、统计信息利用或特定算子优化。

## 延伸阅读

- [DuckDB SQL Introduction](https://duckdb.org/docs/stable/sql/introduction)
- [DuckDB Typecasting](https://duckdb.org/docs/stable/sql/data_types/typecasting)
- [DuckDB EXPLAIN](https://duckdb.org/docs/stable/guides/meta/explain)
