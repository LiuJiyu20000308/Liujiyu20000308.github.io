---
title: "04 · DataChunk 与向量化执行"
order: 4
summary: 从 DataChunk、Vector、SelectionVector 和 UnifiedVectorFormat 出发，理解 DuckDB 如何用批处理摊薄解释开销并改善数据局部性。
keywords: [DataChunk, Vector, SelectionVector, ValidityMask, Vectorized Execution]
description: DuckDB DataChunk、Vector 表示与向量化执行源码笔记。
---

## 为什么既不是一次一行，也不是一次一整列

传统 Volcano 模型让父算子反复调用子算子的 `next()`，每次返回一行。接口清晰，但函数调用、分支判断和表达式分派会落到每一行上。另一端的算子级代码生成能把整条查询融合成机器码，但编译成本和系统复杂度更高。

DuckDB 采用向量化执行：算子一次处理一个固定上限的数据批次。批次足够大，可以摊薄虚函数和调度开销，并让紧凑列数据更适合 CPU Cache 与 SIMD；又足够小，不必物化完整中间结果。核心容器是 `DataChunk`，其中每一列是一个 `Vector`，所有列共享相同的逻辑行数。

在当前阅读的 DuckDB 1.4 系列源码中，默认标准向量大小 `STANDARD_VECTOR_SIZE` 为 2048 行。这是默认批大小，不代表每个 Chunk 永远有 2048 行：最后一批、经过过滤的结果或某些算子输出都可能更小。

## `DataChunk`：执行引擎的数据交换单位

可以把一个 Chunk 想成一个小型列式表：

```text
DataChunk (size = 4)
  column 0, BIGINT : [101, 102, 103, 104]
  column 1, DOUBLE : [8.2, 7.5, NULL, 9.1]
  column 2, VARCHAR: ["A", "B", "A", "C"]
```

Pipeline 中的 Source 产生 Chunk，中间 Operator 接收并转换 Chunk，Sink 消费 Chunk 并更新全局或局部状态。因为接口统一，扫描、Filter、Projection、Join Probe、Aggregate Sink 等算子可以在同一个执行框架中组合。

Chunk 的生命周期通常很短。算子不应默认输入向量在下一次调用后仍然有效；若状态要跨 Chunk 保存数据，需要明确持有或复制相应缓冲区。字符串、List、Struct 等嵌套类型还涉及子向量和额外 Buffer，生命周期错误比基本数值类型更隐蔽。

## Vector 不只有“平铺数组”一种形式

DuckDB 会根据数据特征使用不同 Vector 表示，避免无意义的复制和展开：

- **Flat Vector**：普通连续值数组；
- **Constant Vector**：整个批次共享一个值，例如常量表达式；
- **Dictionary Vector**：通过 SelectionVector 间接引用另一个向量，常用于过滤后的零拷贝切片；
- **Sequence Vector**：用起点与步长表示序列；
- 另外还有针对编码数据或复合类型的表示。

假设 Filter 保留第 `[0, 3, 4]` 行，它不一定立即复制三行数据，而可以让结果向量携带选择索引，继续引用原数据。好处是减少内存搬运；代价是下游算子不能假设输入一定连续。

## `UnifiedVectorFormat` 的作用

如果每个函数都为 Flat、Constant、Dictionary 等表示编写一套循环，代码会迅速膨胀。`ToUnifiedFormat` 将不同表示统一暴露为三部分：

```text
data      → 底层值缓冲区
sel[i]    → 逻辑第 i 行映射到 data 的哪个位置
validity  → 该位置是否为 NULL
```

一个向量化函数的典型循环因此类似：

```cpp
for (idx_t i = 0; i < count; i++) {
    auto row = format.sel->get_index(i);
    if (!format.validity.RowIsValid(row)) {
        // propagate or handle NULL
        continue;
    }
    result[i] = operation(input[row]);
}
```

这不是实际完整代码，而是理解接口的最小模型。性能敏感路径仍会针对常量、无 NULL 或特定物理类型走更快的分支。

## NULL、逻辑类型与物理类型

NULL 通常不作为特殊值塞进数据数组，而由 `ValidityMask` 单独表示。这样数值数组保持紧凑，向量循环也能先走“全有效”的 fast path，只有存在 NULL 时才检查位图。

`LogicalType` 描述 SQL 语义，如 DECIMAL、DATE、VARCHAR；`PhysicalType` 决定内存中使用何种底层表示。多个逻辑类型可能共享同一种物理表示，但比较、Cast 和输出格式仍必须遵循各自语义。编写执行函数时只看 C++ 底层类型而忽略 LogicalType，容易在 Decimal 精度、时间单位或枚举类型上出错。

## 过滤和投影如何流过一个 Chunk

```sql
SELECT price * quantity AS amount
FROM lineitem
WHERE quantity >= 10;
```

可以把一次批处理理解为：Scan 生成 `price`、`quantity` 两个 Vector；Filter 对 `quantity` 批量比较并生成 SelectionVector；Projection 只对保留位置批量执行乘法；结果 Chunk 继续推给 Sink 或 Result Collector。整个过程中不需要为每一行构造通用 Row 对象。

真正的优化点通常是：减少输入列宽、让 Filter 尽早缩小 selection、避免不必要的 Flatten/Copy、为常量与无 NULL 数据提供 fast path，以及让输出 Buffer 可以复用。

## 源码阅读入口

- `src/include/duckdb/common/types/data_chunk.hpp`：Chunk 的列集合、容量与生命周期操作。
- `src/include/duckdb/common/types/vector.hpp`：Vector 与 UnifiedVectorFormat。
- `src/include/duckdb/common/types/selection_vector.hpp`：逻辑行到物理位置的映射。
- `src/include/duckdb/common/types/validity_mask.hpp`：NULL 位图。
- `src/execution/expression_executor/`：表达式的批量执行。
- `src/include/duckdb/common/vector_operations/`：常用向量算子入口。

## 面试时容易被追问

**向量化为什么比逐行执行快？** 它把函数调用和类型分派摊到一批数据上，提高数据局部性，为紧凑循环、SIMD 和特化 fast path 创造条件；并不是“用了 Vector 类”本身就会自动变快。

**Dictionary Vector 为什么既能提速又可能变慢？** 它避免复制，但引入间接寻址。若下游多次随机访问，或 selection 很不连续，Flatten 成连续数据可能反而更合适。

**为什么默认批大小不是越大越好？** 更大的批次能继续摊薄调用开销，却增加工作集、临时内存和缓存压力，也会降低调度与中断响应粒度。批大小是在多种开销之间折中。

## 延伸阅读

- [DuckDB Execution Format](https://duckdb.org/docs/stable/internals/vector)
- [DuckDB Storage Versions and Vector Size](https://duckdb.org/docs/stable/internals/storage)
- [MonetDB/X100: Hyper-Pipelining Query Execution](https://www.cidrdb.org/cidr2005/papers/P19.pdf)
