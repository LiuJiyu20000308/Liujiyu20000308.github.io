---
layout: post
title: 量化开发面经：SMA 行情处理代码优化
date: 2026-8-10 16:30 +0800
tags: [量化开发, C++, 并发, 性能优化]
toc: true
---

> 原始代码：`a.md`
> 场景：单生产者、单消费者处理多合约 Tick，并计算每个合约最近 100 个价格的简单移动平均。
> 说明：本文只分析示例代码本身，不把未经压测的方案写成已经取得的性能结果。

## 1. 面试时先给出的结论

这段代码的线程同步写法基本正确：条件变量使用了谓词，队列与结束标志由同一把锁保护，生产者在解锁后通知消费者，消费者也会在生产结束后把队列排空。真正的问题主要在数据结构和热路径：

1. 每个 Tick 都可能执行 `vector.erase(begin())`，需要搬移窗口内约 99 个 `double`；
2. 每次计算 SMA 都重新遍历 100 个价格求和；
3. 每个 Tick 查询两次 `std::map`，带来 `O(log M)` 查找、指针跳转和节点分配；
4. `instrument_sma_results` 没有任何下游消费者，却仍在持续插入、删除和分配；
5. 共享队列逐 Tick 加锁和通知，且队列无上限，吞吐和过载行为都不理想。

最先应做的不是直接写 lock-free queue，而是把 SMA 改为“环形数组 + 滚动和”，把每 Tick 的窗口维护从 `O(W)` 降到 `O(1)`；再根据合约 ID 的稠密程度，把两个 `map` 合并为一个连续状态数组或 `unordered_map`。随后通过有界队列、批量入队/出队减少同步，并用按合约分片的方式扩展消费者数量。lock-free、绑核和 NUMA 应放在 profiling 证明队列仍是瓶颈之后。

## 2. 原代码实际复杂度

设 Tick 数为 `N`，SMA 窗口为 `W=100`，合约数量为 `M`。

| 操作 | 原实现 | 问题 |
|---|---:|---|
| 查找合约价格历史 | `O(log M)` | `std::map` 节点分散，cache locality 差 |
| 删除最旧价格 | `O(W)` | `vector.erase(begin())` 搬移其余元素 |
| 重新计算 SMA | `O(W)` | 每个成熟窗口都重新遍历 100 个价格 |
| 删除最旧 SMA | `O(W)` | 结果容器也执行一次头删 |
| 队列同步 | 每 Tick 一次锁和通知 | 同步成本可能高于实际计算 |
| 队列空间 | 无界 | 生产者更快时内存可以持续增长 |

因此热路径近似为 `O(N(log M + W))`。`W=100` 看起来不大，但一百万个 Tick 会造成大量重复搬移和加法；而优化后的环形窗口可以做到 `O(N)`。

## 3. 优化项总表

### P0：必须先修的正确性和可运行性问题

| 优化点 | 原代码风险 | 建议 |
|---|---|---|
| 明确结果语义 | `instrument_sma_results` 只保留最近 100 个结果，但程序最后完全不读取 | 若只需要最新指标，每个合约只存一个值；若要保存全量，应流式写入下游，而不是在热路径中维护无用 vector |
| 有界队列与过载策略 | `std::queue` 无上限，消费者落后时会耗尽内存 | 设置容量，并明确阻塞、丢弃、合并更新或落盘中的一种策略 |
| 输入校验 | `num_instruments==0` 会在 `i % num_instruments` 处产生未定义行为 | 启动前校验 Tick 数、合约数和窗口大小 |
| 非法价格 | `NaN` 或无穷价格一旦进入滚动和，后续 SMA 会一直被污染 | 在入口处拒绝或单独标记非有限价格 |
| 时间与顺序语义 | 当前代码按“到达顺序”和“最近 100 笔”计算，不是时间窗口 | 明确是 event time 还是 arrival time，并处理乱序、重复、丢包和 sequence gap |
| 多生产者结束条件 | 一个 `bool` 只适用于单生产者 | 多生产者时使用活跃生产者计数，最后一个生产者负责关闭队列 |
| 异常退出 | 生产者异常退出但没有设置完成标志时，消费者可能永久等待 | 用 RAII 关闭队列，在线程入口捕获异常并通知其他线程停止 |
| 依赖头文件 | 使用 `numeric_limits` 却没有直接包含 `<limits>` | 不依赖其他 STL 头的传递包含，显式包含所用声明的头文件 |

### P1：收益最大且风险最低的热路径优化

#### 3.1 环形数组代替 `erase(begin())`

`vector.erase(begin())` 会把后面所有元素向前搬移。窗口固定为 100 时，更合适的数据结构是固定长度 `std::array<double, 100>`，再用 `next` 指向下一次覆盖的位置：

```cpp
// 窗口未满：在 next 位置写入新价格，count 加一。
// 窗口已满：values[next] 就是即将离开窗口的最旧价格。
// 覆盖后令 next = (next + 1) % Window，整个过程不移动其他元素。
```

这样删除旧值和插入新值都从 `O(W)` 变为 `O(1)`，同时窗口内存连续、大小固定，不再发生动态扩容。

#### 3.2 滚动和代替每次完整求和

窗口未满时执行：

```cpp
sum += new_price;
```

窗口已满时执行：

```cpp
sum += new_price - oldest_price;
```

当 `count == Window` 时，`SMA = sum / Window`。每个 Tick 只需常数次加减，不再重新遍历 100 个价格。

滚动和会累计浮点舍入误差。工程上可以每隔若干次更新重新遍历当前 100 个元素校准一次；这仍是摊还 `O(1)`。如果价格最小变动单位固定，还可以用整数 tick 或定点数保存价格，用整数累计后再转换为 `double`。

#### 3.3 合并两个合约状态容器

价格窗口、滚动和、最新 SMA 本质上属于同一个合约状态，不应分别放在两棵 `map` 中。可以定义：

```cpp
struct InstrumentState {
    RollingSma<100> sma;
    double latest_sma = 0.0;
    bool ready = false;
};
```

如果 `instrument_id` 是 `[0, num_instruments)` 的稠密整数，直接使用：

```cpp
std::vector<InstrumentState> states(num_instruments);
auto& state = states[tick.instrument_id];  // O(1)，连续内存
```

如果 ID 稀疏或取值很大，使用 `std::unordered_map<int, InstrumentState>`，并根据预计合约数提前 `reserve()`。只有确实需要按 ID 有序遍历时才使用 `std::map`。

#### 3.4 删除热路径中的无效结果维护

原代码中的 `instrument_sma_results` 最终没有读取，它产生的 `push_back()`、`erase()`、NaN 写入和内存分配都是无效工作。常见选择是：

- 策略只需要当前指标：每个合约保存一个 `latest_sma`；
- 下游需要每笔结果：计算后立即发送到策略、信号队列或批量持久化模块；
- 只为压测验证正确性：累加到 `checksum`，防止编译器消除计算，不保存全部结果。

#### 3.5 把不变工作移出循环

- `calculate_sma` lambda 没必要在每个 Tick 的循环体内声明；
- `1.0 / SMA_WINDOW_SIZE` 可以预先计算，成熟窗口使用乘法；
- `SMA_WINDOW_SIZE` 应写成 `inline constexpr std::size_t`；
- 对已知数量的动态容器提前 `reserve()`；
- 计数使用 `std::uint64_t` 或 `std::size_t`，避免大规模运行时 `int` 溢出。

### P2：队列与同步优化

#### 3.6 队列必须有界并提供背压

有界队列不仅是节省内存，它定义了系统过载时的正确行为：

- **阻塞生产者**：不丢数据，但行情回调线程可能被拖慢；
- **丢弃最新数据**：保持旧数据完整，但策略看到的数据变旧；
- **丢弃最旧数据**：优先保留最新行情，但 SMA 的统计语义发生变化；
- **按合约合并更新**：适合只关心最新快照，不适合逐笔成交；
- **写入日志再异步回放**：可靠性高，但延迟和工程成本更高。

面试中不能只说“队列满了就丢”，必须先说明业务允许丢什么。

#### 3.7 批量入队和批量出队

原实现每个 Tick 都执行一次加锁、解锁和 `notify_one()`。可以让生产者一次提交一批 Tick，消费者一次取走一批后在锁外计算：

```text
生产者构造 64～512 条本地 batch
  -> 获取一次锁
  -> 整批移动到共享队列
  -> 队列由空变为非空时通知消费者

消费者被唤醒
  -> 获取一次锁
  -> 移走最多 batch_size 条 Tick
  -> 释放锁
  -> 在本地逐条更新 SMA
```

这样锁的次数大致从 `N` 降到 `N / batch_size`。代价是批量会增加等待时间，因此 batch 大小应由吞吐和 P99 延迟共同决定，也可以采用“达到数量或超过几十微秒就刷新”的双阈值。

#### 3.8 保留条件变量写法中的正确部分

原代码以下做法是正确的，不应为了“优化”而改坏：

- `wait(lock, predicate)` 可以正确处理虚假唤醒；
- 队列和 `g_producer_finished` 由同一互斥锁保护，没有数据竞争；
- 必须同时满足“队列为空且生产结束”才能退出，从而保证队列被排空；
- 在解锁后 `notify_one()`/`notify_all()` 可以减少被唤醒线程立刻争抢同一把锁。

手工 `unlock()` 可以改成更小的 RAII 作用域，主要是可读性和异常安全优化，并不是性能关键点。

#### 3.9 什么时候使用 SPSC 无锁环形队列

当前恰好是一名生产者和一名消费者，可以使用固定容量 SPSC ring buffer：生产者只写 `tail`，消费者只写 `head`，双方用 acquire/release 建立数据可见性，不需要互斥锁。

但它不是第一步，原因是：

1. 环形窗口和状态容器通常比换队列更先带来收益；
2. 忙等会持续占用 CPU，休眠又会增加唤醒延迟；
3. 容量、关闭协议、对象生命周期、cache-line padding 和内存序都容易写错；
4. 一旦变成多生产者或多消费者，就不能继续使用简单 SPSC 实现。

更稳妥的顺序是先对 mutex+batch 版本 profiling；只有队列同步仍占主要 CPU 或 P99 不达标时，再换经过充分测试的 SPSC 队列。

### P3：多核扩展与低延迟架构

#### 3.10 不能直接增加消费者线程

SMA 是带状态的计算。同一合约的第 `t` 个 Tick 必须在第 `t-1` 个 Tick 之后更新同一个窗口。如果多个消费者从同一队列随意取 Tick，会出现：

- 同一合约状态被多个线程并发修改，产生数据竞争；
- 即使加锁，也可能因调度顺序不同而打乱该合约的 Tick 顺序；
- 每个消费者若维护自己的局部 map，会把一条合约历史拆散，SMA 直接算错。

正确的扩展方式是按 key 分片：

```cpp
worker_id = stable_hash(instrument_id) % worker_count;
```

同一合约始终进入同一 worker 的有界队列。每个 worker 独占自己的 `InstrumentState`，无需为 SMA 状态加锁，并且可以保留合约内顺序。若上游本身有 sequence number，还应在分片入口检测丢包、重复和乱序。

#### 3.11 数据布局和 cache locality

原结构常见 ABI 下可能占 32 字节，因为 `int instrument_id` 后需要为 `double price` 插入对齐填充。把 8 字节字段放在前面：

```cpp
struct MarketDataTick {
    std::int64_t timestamp;
    double price;
    std::int32_t instrument_id;
    std::int32_t volume;
};
```

常见平台上可缩小到 24 字节，但必须通过本机 `sizeof(MarketDataTick)` 验证，不能把布局大小当成标准保证。

如果 SMA 热路径只读取 `instrument_id` 和 `price`，还可以：

- 将时间戳、成交量放入冷数据结构；
- 批处理中使用 SoA，让价格和 ID 连续存储；
- 让每个 worker 独占一段状态，避免 false sharing；
- profiling 后再考虑 cache-line 对齐、绑核和 NUMA 本地分配。

#### 3.12 减少复制和分配

- 生产者可直接 `emplace`，消费者可移动 Tick；不过当前 Tick 很小，收益需测量；
- 批量 vector 提前 `reserve()` 并循环复用；
- 固定容量 ring 避免队列节点的动态分配；
- 行情接入层尽量解析一次，后续传递轻量结构或稳定引用；
- 不要为了“零拷贝”保存指向上游临时缓冲区的悬空指针，ownership 必须先正确。

### P4：数值正确性和行情语义

#### 3.13 `double`、定点数和滚动误差

`double` 不能精确表示多数十进制价格。只做指标分析时通常可以接受，但要明确误差策略：

- 交易所价格有固定 tick size 时，可把价格转换为整数 tick；
- 滚动和可用 `long double`，或定期完整重算校准；
- 若使用 `double`，比较结果不要直接依赖严格相等；
- 资金、成交金额和账务场景通常更适合定点或 decimal，而不是裸 `double`。

#### 3.14 最近 100 笔不等于最近一段时间

原程序实现的是 tick-count window。活跃合约的 100 笔可能只覆盖几毫秒，冷门合约可能覆盖几分钟。若需求是“最近 10 秒 SMA”，需要按事件时间淘汰：

```text
每个合约保存 (exchange_timestamp, price)
  -> 新 Tick 到来后加入队尾
  -> 删除 timestamp < current_event_time - 10s 的数据
  -> 同步更新滚动和
```

此时还必须定义迟到 Tick 的 watermark、最大乱序容忍度和修正策略。

#### 3.15 行情数据质量

生产环境至少要记录或处理：

- sequence gap、重复包和乱序包；
- 非法 ID、非有限价格、负价格是否允许、异常成交量；
- 交易暂停、复权、换月和合约生命周期；
- 队列丢弃数量、最大深度、处理延迟和数据新鲜度。

这些不一定都写进面试代码，但应在架构回答中主动指出。

### P5：计时、压测和工程质量

#### 3.16 原吞吐数字为什么不能直接作为优化结论

`5.15096e+06 ticks/sec` 只是某一次运行输出，当前计时混合了多种成本：

- 生产者每 Tick 调用一次 `high_resolution_clock::now()`；
- 生成测试数据时执行多次取模；
- 线程启动、调度、共享队列等待和 SMA 计算混在一起；
- Debug/Release、编译器、CPU、频率和系统负载没有记录；
- 只运行一次，没有中位数、方差和尾延迟；
- 结果从未被使用，缺少明确的正确性 checksum。

而且 `high_resolution_clock` 不保证单调，也不保证真实达到纳秒精度。耗时测量使用 `steady_clock`；交易所事件时间则来自行情源，不能用本机高精度时钟代替。

#### 3.17 正确的 benchmark 方法

1. 使用 Release 编译，例如 `-O3 -DNDEBUG -march=native -pthread`；
2. 分开测量纯 SMA kernel、队列传输和端到端链路；
3. 先预热，再重复多轮，报告中位数、P95/P99 和波动范围；
4. 用相同输入回放，验证优化前后每个成熟窗口结果在容差内一致；
5. 累加 checksum 或输出少量采样结果，避免无效基准；
6. 记录 CPU 型号、核心数、编译器、线程亲和性和 queue capacity；
7. 同时观察吞吐、单 Tick 延迟、队列最大深度、丢弃数和 CPU 使用率；
8. 使用 `perf stat`、`perf record` 或火焰图确认热点，而不是凭感觉继续改。

#### 3.18 其他 C++ 工程改进

- 把队列、状态和结束协议封装为类，删除 `g_` 全局变量，使程序可重入、可测试；
- 使用 `std::lock_guard`/`std::unique_lock` 的小作用域表达锁生命周期；
- C++20 可用 `std::jthread` 和 `stop_token` 组织停止流程；
- 多线程日志使用单独日志线程或 `std::osyncstream`，避免输出交错；
- `std::endl` 会强制 flush，非必要日志使用 `\n`；
- 对 `duration.count()==0` 做保护；
- 给 `RollingSma`、队列关闭、队列满和乱序输入编写单元测试；
- 使用 ThreadSanitizer 检查数据竞争，AddressSanitizer/UBSan 检查内存和未定义行为。

## 4. 推荐的优化顺序

不要一次把所有东西都改成复杂的低延迟框架。更可信的工程路径是：

```text
第一步：建立结果对拍和可重复 benchmark
  -> 第二步：环形窗口 + rolling sum
  -> 第三步：map 合并为连续 InstrumentState
  -> 第四步：删除无用结果存储，检查结构体布局和分配
  -> 第五步：有界队列 + batch，测吞吐与 P99
  -> 第六步：按 instrument 分片到多个 worker
  -> 第七步：若 profiling 仍指向同步，再评估 SPSC/绑核/NUMA
```

这个顺序的好处是每一步都能独立验证正确性和收益，也能在性能回退时快速定位是哪一层引入的问题。

## 5. 一个可运行的优化版本

下面的版本保留“单生产者、单消费者、最近 100 笔 SMA”的原始语义，重点演示：

- 固定数组和滚动和；
- 连续的合约状态；
- 有界批量队列；
- 锁外批量计算；
- 非法输入检查、队列关闭和 checksum。

它不是宣称适用于所有生产行情系统的最终版本；真正上线前还要接入 sequence、丢包策略、监控和异常传播。

```cpp
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

struct MarketDataTick {
    std::int64_t timestamp;
    double price;
    std::int32_t instrument_id;
    std::int32_t volume;
};

template <std::size_t Window>
class RollingSma {
    static_assert(Window > 0, "SMA window must be positive");

public:
    std::optional<double> update(double price) {
        // 1. 未满时直接累加；已满时先从 sum 中扣掉即将被覆盖的最旧值。
        if (count_ < Window) {
            values_[next_] = price;
            sum_ += price;
            ++count_;
        } else {
            sum_ += price - values_[next_];
            values_[next_] = price;
        }

        // 2. next_ 始终指向下一次写入位置，因此数组无需 erase 或搬移。
        next_ = (next_ + 1) % Window;

        // 3. 滚动和长期更新会积累舍入误差，低频完整重算进行校准。
        if (--updates_until_rebase_ == 0) {
            sum_ = 0.0;
            for (std::size_t i = 0; i < count_; ++i) {
                sum_ += values_[i];
            }
            updates_until_rebase_ = kRebaseInterval;
        }

        // 4. 窗口未满时没有 SMA，使用 optional 明确表达，而不是写入 NaN 占位。
        if (count_ < Window) {
            return std::nullopt;
        }
        return sum_ * kInverseWindow;
    }

private:
    static constexpr std::size_t kRebaseInterval = 4096;
    static constexpr double kInverseWindow = 1.0 / static_cast<double>(Window);

    std::array<double, Window> values_{};
    std::size_t next_ = 0;
    std::size_t count_ = 0;
    std::size_t updates_until_rebase_ = kRebaseInterval;
    double sum_ = 0.0;
};

template <typename T>
class BoundedBatchQueue {
public:
    explicit BoundedBatchQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("queue capacity must be positive");
        }
    }

    bool push_batch(std::vector<T>& batch) {
        if (batch.empty()) {
            return true;
        }
        if (batch.size() > capacity_) {
            throw std::invalid_argument("batch is larger than queue capacity");
        }

        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] {
            return closed_ || queue_.size() + batch.size() <= capacity_;
        });
        if (closed_) {
            return false;
        }

        const bool was_empty = queue_.empty();
        queue_.insert(queue_.end(),
                      std::make_move_iterator(batch.begin()),
                      std::make_move_iterator(batch.end()));
        batch.clear();
        lock.unlock();

        // 消费者已经在运行且队列非空时无需为每一批重复唤醒。
        if (was_empty) {
            not_empty_.notify_one();
        }
        return true;
    }

    std::size_t pop_batch(std::vector<T>& output, std::size_t max_batch_size) {
        if (max_batch_size == 0) {
            throw std::invalid_argument("max batch size must be positive");
        }

        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });

        // closed 且队列为空，表示所有已有 Tick 都已经被排空。
        if (queue_.empty()) {
            output.clear();
            return 0;
        }

        const std::size_t count = std::min(max_batch_size, queue_.size());
        output.clear();
        for (std::size_t i = 0; i < count; ++i) {
            output.push_back(std::move(queue_.front()));
            queue_.pop_front();
        }
        lock.unlock();
        not_full_.notify_one();
        return count;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    const std::size_t capacity_;
    std::deque<T> queue_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};

template <std::size_t Window>
class SmaEngine {
public:
    explicit SmaEngine(std::size_t instrument_count)
        : states_(instrument_count) {
        if (instrument_count == 0) {
            throw std::invalid_argument("instrument count must be positive");
        }
    }

    std::optional<double> on_tick(const MarketDataTick& tick) {
        if (tick.instrument_id < 0 ||
            static_cast<std::size_t>(tick.instrument_id) >= states_.size()) {
            throw std::out_of_range("invalid instrument id");
        }
        if (!std::isfinite(tick.price)) {
            return std::nullopt;  // 生产系统中还应增加错误计数和告警。
        }
        return states_[static_cast<std::size_t>(tick.instrument_id)].update(tick.price);
    }

private:
    std::vector<RollingSma<Window>> states_;
};

int main() {
    constexpr std::size_t kWindow = 100;
    constexpr std::size_t kBatchSize = 256;
    constexpr std::size_t kQueueCapacity = 1U << 16;
    constexpr std::size_t kTickCount = 1'000'000;
    constexpr std::size_t kInstrumentCount = 100;

    BoundedBatchQueue<MarketDataTick> queue(kQueueCapacity);
    double checksum = 0.0;
    std::uint64_t ready_count = 0;

    const auto start = std::chrono::steady_clock::now();

    std::thread producer([&] {
        std::vector<MarketDataTick> batch;
        batch.reserve(kBatchSize);

        for (std::size_t i = 0; i < kTickCount; ++i) {
            const auto id = static_cast<std::int32_t>(i % kInstrumentCount);
            batch.push_back(MarketDataTick{
                static_cast<std::int64_t>(i),  // 回放基准使用确定性事件时间。
                100.0 + static_cast<double>(i % 100) * 0.1 +
                    static_cast<double>(i % 7) * 0.05,
                id,
                static_cast<std::int32_t>(100 + i % 50)
            });

            if (batch.size() == kBatchSize) {
                queue.push_batch(batch);
            }
        }
        queue.push_batch(batch);  // 提交最后不足一批的数据。
        queue.close();
    });

    std::thread consumer([&] {
        SmaEngine<kWindow> engine(kInstrumentCount);
        std::vector<MarketDataTick> batch;
        batch.reserve(kBatchSize);

        while (queue.pop_batch(batch, kBatchSize) != 0) {
            // 队列锁已释放，整批计算不会阻塞生产者。
            for (const MarketDataTick& tick : batch) {
                if (auto sma = engine.on_tick(tick)) {
                    checksum += *sma;
                    ++ready_count;
                }
            }
        }
    });

    producer.join();
    consumer.join();

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double throughput = static_cast<double>(kTickCount) / elapsed.count();

    std::cout << std::fixed << std::setprecision(3)
              << "processed=" << kTickCount
              << " ready=" << ready_count
              << " seconds=" << elapsed.count()
              << " ticks_per_second=" << throughput
              << " checksum=" << checksum << '\n';
}
```

### 这版代码仍有哪些边界

1. 批量生产会等待凑满 `kBatchSize`，真实行情接入应增加时间阈值刷新；
2. `deque` 仍可能分配内存，极低延迟场景可换固定容量 ring；
3. 示例只有一个消费者，扩展时必须按 `instrument_id` 分片；
4. 线程异常传播、停止令牌和监控指标仍需补齐；
5. checksum 只防止计算被忽略，不能代替逐 Tick 结果对拍；
6. 这个 benchmark 同时测量数据生成、队列和计算，仍应另测纯 kernel。

## 6. 优化前后核心差异

| 维度 | 原实现 | 优化版本 |
|---|---|---|
| 窗口 | `vector` 头删 | 固定数组环形覆盖 |
| SMA | 每 Tick 遍历窗口 | `sum += new-old` |
| 合约状态 | 两个 `map` | 一个连续 `vector` |
| 结果 | 保存但不使用的 vector | 最新结果/流式下游/checksum |
| 同步 | 每 Tick 锁和通知 | 有界批量入队、批量出队 |
| 过载 | 内存无限增长 | 明确容量和背压 |
| 计时 | `high_resolution_clock` 混合测量 | `steady_clock`，并建议拆分 benchmark |
| 数值 | 无非法值与误差策略 | 有限值检查、周期校准，可进一步定点化 |
| 扩展 | 不能安全地直接加消费者 | 按合约 ID 分片到单写 worker |

## 7. 面试精简版回答

> 我会先 profile，但从代码上能直接看到三个主要热点。第一，`vector.erase(begin())` 和每次重新遍历 100 个价格，让单 Tick 窗口更新是 O(W)；我会改成固定环形数组和 rolling sum，把它降到 O(1)。第二，当前每个 Tick 要查两次 `std::map`，而合约 ID 是连续的，可以把价格窗口、sum 和最新结果合并成一个 `InstrumentState vector`，减少 O(log M) 查找和指针跳转。第三，队列逐 Tick 加锁通知且没有容量，我会先做有界队列和批处理，明确背压，再看 profiling 是否值得换 SPSC ring。若要多核扩展，不能让多个消费者随便抢同一个队列，而要按 instrument_id 分片，保证同一合约由一个 worker 按序更新。

## 8. 面试追问

### Q1：为什么不一开始就写 lock-free queue？

因为 lock-free 只解决传输同步，不会消除原来 `O(W)` 的窗口搬移、求和和两次 map 查找，而且关闭协议、内存序、容量和忙等策略更容易出错。我会先做低风险算法与布局优化，再通过 profiling 判断 mutex 是否仍是瓶颈。

### Q2：rolling sum 会不会不准确？

会有长期浮点误差积累，所以可以每隔固定次数对窗口做一次完整求和校准；价格有固定最小变动单位时，也可以转成整数 tick 累加。账务金额不能直接照搬 `double` 方案。

### Q3：为什么多消费者会算错？

SMA 是每个合约的有序状态机。两个线程同时处理同一合约会产生数据竞争或乱序。应该按 instrument_id 做稳定分片，让一个合约始终归一个 worker；不同合约之间才能安全并行。

### Q4：batch 越大越好吗？

不是。batch 大会摊薄锁开销并提高吞吐，但也会增加 Tick 等待凑批的时间，使 P99 延迟变差。实际应使用数量和时间双阈值，并通过吞吐—延迟曲线选择参数。

### Q5：如何证明优化没有改错？

固定一份输入回放，让朴素完整求和作为 reference，与 rolling sum 对每个合约、每个成熟窗口逐点比较；浮点结果使用合理容差。同时覆盖窗口从 99 到 100、重复价格、非法价格、队列关闭、队列满和乱序数据等边界测试。

### Q6：吞吐高就代表低延迟吗？

不代表。批处理可能提高总吞吐，却让单个 Tick 等待更久。因此至少同时报告 ticks/s、端到端 P50/P95/P99、最大队列深度、数据新鲜度和丢弃数量。

## 9. 最后检查清单

- [ ] 环形窗口替代头删；
- [ ] rolling sum 替代每 Tick 完整求和；
- [ ] 状态合并并根据 ID 稠密性选择 vector/unordered_map；
- [ ] 删除没有消费者的 SMA 结果存储；
- [ ] 有界队列、背压与丢弃语义明确；
- [ ] 批量传输在锁外计算；
- [ ] 多 worker 按合约稳定分片；
- [ ] 处理非法值、重复、丢包、乱序和 sequence gap；
- [ ] 区分 Tick 数量窗口与时间窗口；
- [ ] 使用 steady clock 和可重复回放；
- [ ] 优化前后逐点对拍；
- [ ] 同时测吞吐、尾延迟、队列深度和 CPU；
- [ ] 使用 profiler、perf 和 sanitizer 验证，而不是只凭代码直觉。
