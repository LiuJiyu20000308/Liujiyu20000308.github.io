---
layout: post
title: 量化开发面经：从原始代码逐版优化 SMA 行情处理
date: 2026-8-10 16:05 +0800
tags: [量化开发, C++, 并发, 性能优化]
toc: true
---

> 原始代码：`a.md`
> 场景：单生产者、单消费者处理多个合约的 Tick，并计算每个合约最近 100 笔价格的 SMA。
> 原则：每一版只解决一层问题。先保证能解释、能验证，再进入更复杂的并发优化。

## 0. 先看演进路线

这段代码不应该一步改成 lock-free、多 worker 的最终架构。面试中更合理的回答顺序是：

```text
原始版本
  每个 Tick：map 查找 + vector 头删 + 遍历窗口求和 + 单条队列同步

第一版：只改 SMA 计算
  保留 map、mutex、condition_variable 和单生产者/单消费者
  vector -> deque，完整求和 -> rolling sum

第二版：只改合约状态和内存布局
  map -> vector，deque -> 固定长度环形数组
  去掉节点查找、动态分配和指针跳转

第三版：只改线程间传输
  无界逐 Tick 队列 -> 有界队列 -> 批量入队/出队
  明确背压，同时减少锁和唤醒次数

第四版：再做多核扩展
  按 instrument_id 稳定分片，每个合约只由一个 worker 更新

第五版：profiling 证明有必要后再做
  SPSC ring、绑核、cache-line padding、NUMA、hot/cold split
```

下面每一版都先说明“相对上一版改了什么”，没有提到的部分保持不变。

## 1. 原始版本：先准确说出问题

原代码的并发正确性基础并不差：

- `wait(lock, predicate)` 能处理虚假唤醒；
- 队列和 `g_producer_finished` 由同一把 mutex 保护；
- 消费者只有在“队列为空且生产者结束”时退出，因此不会遗漏队列里已有的数据；
- 生产者在释放锁后通知消费者，可以避免被唤醒线程立刻争抢同一把锁。

第一步不是把这些全部推翻，而是先找 consumer 热路径中确定存在的重复工作。

### 1.1 每个成熟 Tick 实际做了什么

原代码对一个已经填满 100 个价格的合约执行：

```cpp
instrument_price.erase(instrument_price.begin()); // 搬移其余 99 个 double
instrument_price.push_back(current_tick.price);
instrument_sma.erase(instrument_sma.begin());      // 再搬移一遍结果
instrument_sma.push_back(calculate_sma(instrument_price));
                                                    // 再遍历 100 个价格
```

此外，每个 Tick 还通过两个 `std::map::operator[]` 分别寻找价格和结果：

```cpp
instrument_sma_results[current_tick.instrument_id];
instrument_history_prices[current_tick.instrument_id];
```

设 Tick 数为 `N`、窗口大小为 `W`、合约数为 `M`，原热路径近似为：

```text
O(N * (log M + W))
```

### 1.2 明确本题的历史数据语义

这里的需求是每个合约同时保留两段历史：

1. 最近 100 个 price；
2. 最近 100 个与 Tick 一一对应的 SMA 结果。

每收到一个有效 price，都必须产生一条 SMA 记录：

- 当前 price 历史不足 100 个时，SMA 记录为 `NaN`；
- 收到第 100 个 price 时，第一次得到有效 SMA；
- 第 101 个 price 到来后，淘汰最旧 price，并计算新的 SMA；
- SMA 历史同样只保留最近 100 条，淘汰规则与 Tick 顺序对齐。

因此不能像上一稿那样只保存 `latest_sma`。下面每个版本都保留 100 个 price 和 100 个 SMA；checksum 只用于 benchmark 校验，不替代 SMA 历史。

## 2. 第一版：只改 SMA 计算，其他架构全部保留

### 2.1 相对原代码只改三件事

1. `vector<double>` 改为 `deque<double>`，使用 `pop_front()` 删除最旧价格；
2. 每个合约保存 `sum`，新价格加入时加上，旧价格离开时减掉；
3. SMA 结果也改成 `deque`，每个 Tick 写入一条，并只保留最近 100 条。

以下部分故意不改：

- 仍使用 `std::map`；
- 仍使用全局 `std::queue`；
- 仍然逐 Tick 加锁、出队；
- 仍然只有一个生产者和一个消费者。

这样能把“算法优化”和“并发架构优化”的收益分开验证。

### 2.2 第一版完整代码

```cpp
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

struct MarketDataTick {
    long long timestamp;
    int instrument_id;
    double price;
    int volume;

    MarketDataTick(long long ts = 0, int id = 0,
                   double p = 0.0, int vol = 0)
        : timestamp(ts), instrument_id(id), price(p), volume(vol) {}
};

std::queue<MarketDataTick> g_tick_queue;
std::mutex g_queue_mutex;
std::condition_variable g_queue_cv;
bool g_producer_finished = false;

inline constexpr std::size_t kSmaWindowSize = 100;

// 第一版的新状态：仍由 map 保存，但窗口使用 deque，并额外维护 rolling sum。
struct InstrumentSmaState {
    std::deque<double> prices;
    std::deque<double> sma_history;
    double sum = 0.0;
};

void producer(int num_ticks, int num_instruments) {
    for (int i = 0; i < num_ticks; ++i) {
        MarketDataTick tick(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count(),
            i % num_instruments,
            100.0 + (i % 100) * 0.1 + (i % 7) * 0.05,
            100 + (i % 50)
        );

        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            g_tick_queue.push(tick);
        }
        g_queue_cv.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_producer_finished = true;
    }
    g_queue_cv.notify_all();
}

void consumer() {
    // 与原代码一样仍使用 map；第一版不碰这层结构。
    std::map<int, InstrumentSmaState> states;

    std::uint64_t processed_count = 0;
    std::uint64_t sma_output_count = 0;
    std::uint64_t nan_count = 0;
    std::uint64_t ready_count = 0;
    double checksum = 0.0;

    while (true) {
        MarketDataTick tick;

        // 出队逻辑与原代码相同，只把手动 unlock 改成更清晰的 RAII 作用域。
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_queue_cv.wait(lock, [] {
                return !g_tick_queue.empty() || g_producer_finished;
            });

            if (g_tick_queue.empty() && g_producer_finished) {
                break;
            }

            tick = g_tick_queue.front();
            g_tick_queue.pop();
        }

        InstrumentSmaState& state = states[tick.instrument_id];

        // 步骤 1：新价格进入窗口，同时加入 rolling sum。
        state.prices.push_back(tick.price);
        state.sum += tick.price;

        // 步骤 2：窗口超过 100 时，减去并删除最旧价格。
        if (state.prices.size() > kSmaWindowSize) {
            state.sum -= state.prices.front();
            state.prices.pop_front();
        }

        // 步骤 3：先把本 Tick 的结果初始化为 NaN。
        // price 不足 100 个时，NaN 就是本 Tick 应保存和返回的结果。
        double current_sma = std::numeric_limits<double>::quiet_NaN();

        // 步骤 4：只有 price 窗口恰好填满后才计算有效 SMA。
        if (state.prices.size() == kSmaWindowSize) {
            current_sma = state.sum / static_cast<double>(kSmaWindowSize);
            checksum += current_sma;
            ++ready_count;
        } else {
            ++nan_count;
        }

        // 步骤 5：每个 Tick 都保存一条 SMA；历史最多保留最近 100 条。
        state.sma_history.push_back(current_sma);
        if (state.sma_history.size() > kSmaWindowSize) {
            state.sma_history.pop_front();
        }

        ++sma_output_count;
        ++processed_count;
    }

    std::size_t retained_price_count = 0;
    std::size_t retained_sma_count = 0;
    for (const auto& entry : states) {
        retained_price_count += entry.second.prices.size();
        retained_sma_count += entry.second.sma_history.size();
    }

    std::cout << "processed=" << processed_count
              << " sma_outputs=" << sma_output_count
              << " nan=" << nan_count
              << " ready=" << ready_count
              << " retained_price=" << retained_price_count
              << " retained_sma=" << retained_sma_count
              << " checksum=" << checksum << '\n';
}

int main() {
    constexpr int kTickCount = 1'000'000;
    constexpr int kInstrumentCount = 100;

    static_assert(kInstrumentCount > 0);

    std::thread producer_thread(producer, kTickCount, kInstrumentCount);
    std::thread consumer_thread(consumer);
    producer_thread.join();
    consumer_thread.join();
}
```

### 2.3 第一版为什么已经有明显改进

成熟窗口每个 Tick 的计算从：

```text
移动 99 个价格 + 移动 99 个 SMA + 遍历 100 个价格
```

变成：

```text
price push/pop + sum 加新值/减旧值 + SMA history push/pop
```

窗口维护从 `O(W)` 降为摊还 `O(1)`。整个 consumer 仍包含 `map` 查询，因此总体近似为：

```text
O(N log M)
```

### 2.4 第一版仍然没有解决什么

- `map` 仍然是树结构，每个 Tick 有 `O(log M)` 查找和指针跳转；
- `deque` 内部仍可能分配分段存储；
- 共享队列仍是无界的；
- 生产者和消费者仍逐 Tick 争用 mutex；
- 多消费者仍不能安全扩展；
- `double` rolling sum 仍会逐渐积累舍入误差；
- 还没有封装“每次 update 必须返回一个数，未成熟时返回 NaN”的接口。

这些问题留到后续版本，不在第一版一次解决。

### 2.5 第一版怎么验证

用原始“每次遍历窗口求和”的实现作为 reference，对固定输入逐 Tick 比较：

```cpp
const double error = std::abs(rolling_sma - reference_sma);
if (error > 1e-10) {
    // 报告 instrument_id、tick 序号和两个结果。
}
```

100 个合约均匀接收 1,000,000 个 Tick 时，每个合约收到 10,000 个 Tick，前 99 个没有 SMA，因此成熟结果数应为：

```text
100 * (10000 - 99) = 990100
```

同时还应满足：

```text
SMA 输出总数 = 1,000,000
NaN 输出总数 = 100 * 99 = 9,900
最终保留的 SMA 历史数 = 100 个合约 * 每个 100 条 = 10,000
```

## 3. 第二版：只改合约状态和内存布局

第一版已经消除了最明显的 `O(W)` 重复工作。第二版再解决 `map + deque` 的查找、分配和 cache locality，但仍保留原来的全局 mutex 队列。

### 3.1 相对第一版改什么

1. 合约 ID 已知是 `[0, num_instruments)`，所以 `map` 改为 `vector`；
2. price 和 SMA 历史都固定为 100，所以两个 `deque` 都改为 `std::array<double, 100>`；
3. 两段历史分别使用 `next_` 指向下一次覆盖位置，形成真正的环形窗口；
4. 每隔一段时间完整重算窗口和，限制浮点累计误差。

队列、生产者、结束标志和单消费者仍保持第一版不变。

### 3.2 固定长度环形 SMA

```cpp
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

// 通用的固定长度历史：写满后覆盖最旧元素，始终只保留最近 Capacity 条。
template <typename T, std::size_t Capacity>
class FixedHistory {
    static_assert(Capacity > 0);

public:
    void push(const T& value) {
        values_[next_] = value;
        next_ = (next_ + 1) % Capacity;
        if (count_ < Capacity) {
            ++count_;
        }
    }

    std::size_t size() const noexcept {
        return count_;
    }

    // 按“从旧到新”的逻辑顺序读取，而不是按 array 的物理下标读取。
    const T& oldest_at(std::size_t offset) const {
        if (offset >= count_) {
            throw std::out_of_range("history offset out of range");
        }
        const std::size_t oldest = count_ < Capacity ? 0 : next_;
        return values_[(oldest + offset) % Capacity];
    }

private:
    std::array<T, Capacity> values_{};
    std::size_t next_ = 0;
    std::size_t count_ = 0;
};

template <std::size_t Window>
class RollingSma {
    static_assert(Window > 0);

public:
    double update(double price) {
        if (count_ < Window) {
            // 窗口未满：next_ 指向尚未使用的位置。
            values_[next_] = price;
            sum_ += price;
            ++count_;
        } else {
            // 窗口已满：values_[next_] 就是最旧值。
            sum_ += price - values_[next_];
            values_[next_] = price;
        }

        // 环形移动，不再删除或搬移其他元素。
        next_ = (next_ + 1) % Window;

        // rolling sum 会积累舍入误差，低频校准仍是摊还 O(1)。
        if (--updates_until_rebase_ == 0) {
            sum_ = 0.0;
            for (std::size_t i = 0; i < count_; ++i) {
                sum_ += values_[i];
            }
            updates_until_rebase_ = kRebaseInterval;
        }

        // 每个 price 都返回一个结果；不足 Window 时按需求返回 NaN。
        if (count_ < Window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return sum_ * kInverseWindow;
    }

    std::size_t size() const noexcept {
        return count_;
    }

private:
    static constexpr std::size_t kRebaseInterval = 4096;
    static constexpr double kInverseWindow =
        1.0 / static_cast<double>(Window);

    std::array<double, Window> values_{};
    std::size_t next_ = 0;
    std::size_t count_ = 0;
    std::size_t updates_until_rebase_ = kRebaseInterval;
    double sum_ = 0.0;
};

template <std::size_t Window>
class InstrumentState {
public:
    double on_price(double price) {
        const double sma = price_window_.update(price);
        sma_history_.push(sma);  // NaN 和有效 SMA 都进入最近 100 条历史。
        return sma;
    }

    double on_invalid_price() {
        // 非法 price 不污染 price window，但当前 Tick 仍留下一个 NaN 结果。
        const double nan = std::numeric_limits<double>::quiet_NaN();
        sma_history_.push(nan);
        return nan;
    }

    const FixedHistory<double, Window>& sma_history() const noexcept {
        return sma_history_;
    }

    std::size_t price_history_size() const noexcept {
        return price_window_.size();
    }

private:
    RollingSma<Window> price_window_;            // 最近 100 个有效 price。
    FixedHistory<double, Window> sma_history_;  // 最近 100 个对应输出。
};
```

### 3.3 consumer 只替换状态部分

原来的出队代码完全不动，只将 consumer 中的状态定义和 Tick 处理替换为：

```cpp
void consumer(std::size_t instrument_count) {
    // ID 稠密时，states[id] 比 map 查找更直接且内存连续。
    std::vector<InstrumentState<100>> states(instrument_count);

    std::uint64_t processed_count = 0;
    std::uint64_t sma_output_count = 0;
    std::uint64_t nan_count = 0;
    std::uint64_t ready_count = 0;
    double checksum = 0.0;

    while (true) {
        MarketDataTick tick;

        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_queue_cv.wait(lock, [] {
                return !g_tick_queue.empty() || g_producer_finished;
            });
            if (g_tick_queue.empty() && g_producer_finished) {
                break;
            }
            tick = g_tick_queue.front();
            g_tick_queue.pop();
        }

        if (tick.instrument_id < 0 ||
            static_cast<std::size_t>(tick.instrument_id) >= states.size()) {
            continue;
        }

        auto& state = states[static_cast<std::size_t>(tick.instrument_id)];
        const double sma = std::isfinite(tick.price)
            ? state.on_price(tick.price)
            : state.on_invalid_price();

        // NaN 是合法的 warm-up 输出，但不进入数值 checksum。
        if (!std::isnan(sma)) {
            checksum += sma;
            ++ready_count;
        } else {
            ++nan_count;
        }
        ++sma_output_count;
        ++processed_count;
    }
}
```

### 3.4 为什么不总是用 vector

这里能够使用 `vector`，是因为模拟数据明确生成连续 ID：

```cpp
i % num_instruments
```

真实市场的 instrument ID 如果非常稀疏，直接按最大 ID 创建 vector 会浪费大量空间。此时第二版应改为：

```cpp
std::unordered_map<int, InstrumentState<100>> states;
states.reserve(expected_instrument_count);
```

只有需要按合约 ID 有序遍历时才保留 `std::map`。

### 3.5 第二版达到的复杂度

每个 Tick：

- 按下标定位状态：`O(1)`；
- 覆盖环形窗口：`O(1)`；
- 更新滚动和：`O(1)`；
- 周期校准：摊还 `O(1)`。

计算部分整体降为 `O(N)`。这时 profiling 很可能开始显示共享队列同步占比上升，才进入第三版。

## 4. 第三版：只改队列，不改变 SMA 算法

第二版的计算已经很轻，逐 Tick mutex、条件变量通知和无界 `std::queue` 可能成为主要问题。第三版分两小步，先保证过载正确，再优化吞吐。

### 4.1 第三版 A：先把队列变成有界队列

原队列无上限。如果生产者持续快于消费者，内存会一直增长。增加容量后，队列满时必须定义行为：

| 策略 | 优点 | 风险 |
|---|---|---|
| 阻塞生产者 | 不丢 Tick | 可能阻塞行情回调线程 |
| 丢最新 Tick | 保住已有队列 | 数据越来越旧 |
| 丢最旧 Tick | 保留最新行情 | 破坏逐笔 SMA 语义 |
| 同合约只保留最新值 | 适合快照 | 不适合逐笔成交 |
| 写日志后回放 | 可靠 | 延迟和工程成本更高 |

示例先选择“阻塞生产者”，因为它最容易保持与原代码相同的结果语义。实际系统必须根据行情类型决定。

### 4.2 第三版 B：在有界基础上批量传输

```cpp
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <vector>

template <typename T>
class BoundedBatchQueue {
public:
    explicit BoundedBatchQueue(std::size_t capacity)
        : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument("capacity must be positive");
        }
    }

    bool push_batch(std::vector<T>& batch) {
        if (batch.empty()) {
            return true;
        }
        if (batch.size() > capacity_) {
            throw std::invalid_argument("batch exceeds queue capacity");
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

        // 只有从空变为非空时，等待中的消费者才一定需要被唤醒。
        if (was_empty) {
            not_empty_.notify_one();
        }
        return true;
    }

    std::size_t pop_batch(std::vector<T>& output,
                          std::size_t max_batch_size) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            output.clear();
            return 0;  // closed 且已经排空。
        }

        const std::size_t count =
            std::min(max_batch_size, queue_.size());
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
```

### 4.3 生产者如何逐步接入 batch

```cpp
constexpr std::size_t kBatchSize = 256;
std::vector<MarketDataTick> batch;
batch.reserve(kBatchSize);

for (std::size_t i = 0; i < tick_count; ++i) {
    batch.push_back(make_tick(i));

    if (batch.size() == kBatchSize) {
        queue.push_batch(batch);  // 一批只获取一次锁。
    }
}

queue.push_batch(batch);  // 提交最后不足 256 条的数据。
queue.close();
```

真实行情不能一直等到凑够 256 条，应使用双阈值：

```text
batch.size() 达到上限
    或
距离上次提交超过 latency_budget
```

满足任一条件就提交，避免低流量时 Tick 长时间滞留在生产者本地。

### 4.4 消费者如何在锁外计算

```cpp
std::vector<MarketDataTick> batch;
batch.reserve(kBatchSize);

while (queue.pop_batch(batch, kBatchSize) != 0) {
    // pop_batch 返回时已经释放队列锁。
    for (const MarketDataTick& tick : batch) {
        const double sma = states[tick.instrument_id].on_price(tick.price);
        // state 内部已经把本次结果写入最近 100 条 SMA 历史。
        // warm-up 阶段的 NaN 不参与 checksum，但仍是正式输出。
        if (!std::isnan(sma)) {
            checksum += sma;
        }
    }
}
```

锁的次数大致从 `N` 降到 `N / batch_size`。但 batch 越大不一定越好：吞吐通常提高，单 Tick 等待时间和 P99 延迟也可能增加。

### 4.5 第三版必须新增的测试

- 队列容量为 1；
- 生产者比消费者快，确认不会无限增长；
- 消费者等待时关闭队列，确认能够退出；
- 队列关闭前已有数据，确认先排空再退出；
- 最后一批不足 batch size，确认没有遗漏；
- 优化前后 ready count 和逐 Tick SMA 完全对应；
- 记录最大 queue depth、生产者阻塞时间和 P99 延迟。

## 5. 第四版：按合约分片到多个 worker

只有单消费者 CPU 已经成为瓶颈时，才进入第四版。

### 5.1 为什么不能让多个消费者直接抢同一个队列

SMA 是每个合约的有序状态机。同一合约的第 `t` 个 Tick 必须在第 `t-1` 个 Tick 之后更新同一窗口。多个消费者自由竞争会产生三类问题：

1. 多线程同时修改同一状态，产生 data race；
2. 即使给状态加锁，获得锁的先后也可能打乱 Tick 顺序；
3. 每个 worker 如果维护独立 states，同一合约历史会被拆散。

### 5.2 正确方案：按 instrument_id 稳定路由

```cpp
std::size_t shard_for(int instrument_id, std::size_t worker_count) {
    return std::hash<int>{}(instrument_id) % worker_count;
}

void dispatch(const MarketDataTick& tick) {
    const std::size_t worker = shard_for(tick.instrument_id, workers.size());
    workers[worker].queue.push(tick);
}
```

架构变成：

```text
行情接入/dispatcher
  -> instrument hash
     -> worker 0 queue -> worker 0 独占 states
     -> worker 1 queue -> worker 1 独占 states
     -> worker 2 queue -> worker 2 独占 states
     -> worker 3 queue -> worker 3 独占 states
```

同一合约永远进入同一 worker，因此：

- 合约内顺序可以保留；
- SMA 状态只有一个写线程，不需要加锁；
- 不同合约可以并行；
- 每个 worker 可以独立输出 batch，减少共享结果锁。

### 5.3 第四版还要考虑负载倾斜

简单 hash 只能平衡合约数量，不能保证 Tick 数均衡。一个极活跃合约仍可能让某个 worker 过载。可选方法：

- 根据历史 Tick 速率做 weighted assignment；
- 定期在安全点迁移整个合约状态；
- 对极热合约单独分配 worker；
- 监控每个 shard 的吞吐、queue depth 和 P99。

不能把同一合约的单个 SMA 状态随意拆给多个线程，因为状态转移本身有严格顺序。

## 6. 第五版：profiling 后才考虑低延迟专项优化

前四版完成后，再用 `perf` 或火焰图确认剩余瓶颈。如果同步和 cache miss 仍占主要成本，才评估下列优化。

### 6.1 每个 shard 使用 SPSC ring

dispatcher 是一个生产者，每个 worker 是自己队列的唯一消费者，满足 SPSC 条件。固定容量 ring 可以避免 mutex 和队列节点分配。

核心思路是：

```text
producer 只更新 tail
consumer 只更新 head
producer 发布新元素时使用 release
consumer 读取 tail 时使用 acquire
```

但必须同时设计：

- 队列满时的背压/丢弃语义；
- 关闭协议；
- head 与 tail 的 cache-line padding；
- 忙等、yield、睡眠或 `atomic_wait` 的切换策略；
- 元素构造、析构与异常安全。

因此不建议面试现场把“自己手写 lock-free”说成第一选择，可以说优先使用经过验证的队列实现。

### 6.2 Tick 结构体布局

原结构顺序在常见 ABI 上可能产生填充：

```cpp
struct MarketDataTick {
    long long timestamp;
    int instrument_id;
    double price;
    int volume;
};
```

可以把 8 字节字段放在一起：

```cpp
struct MarketDataTick {
    std::int64_t timestamp;
    double price;
    std::int32_t instrument_id;
    std::int32_t volume;
};
```

常见平台上可能从 32 字节缩小到 24 字节，但这不是 C++ 标准保证，必须用本机 `sizeof` 和 benchmark 验证。

若 SMA 热路径只读取 ID 和价格，可以进一步把 timestamp、volume 放入冷数据，或批量使用 SoA：

```text
instrument_ids[]
prices[]
timestamps[]
volumes[]
```

### 6.3 绑核、NUMA 和 false sharing

- dispatcher 与 worker 可绑定独立 CPU，降低迁核和 cache 抖动；
- 队列内存和合约状态在消费它的 NUMA node 上分配；
- 不同 worker 的高频计数器分开 cache line，避免 false sharing；
- 线程数不要超过实际可用核心，并避免与 MKL/OpenMP 内部线程过度订阅。

这些优化高度依赖硬件，必须在确定的部署环境中测试，不能只凭代码推断收益。

## 7. 每一版的变化汇总

| 版本 | 只解决什么 | 保留什么 | 计算复杂度 |
|---|---|---|---:|
| 原始版 | 基本生产消费链路 | `map + vector + 完整求和` | `O(N(log M+W))` |
| 第一版 | price/SMA 头删和重复求和 | `map + 两段 100 条历史 + 原队列` | `O(N log M)` |
| 第二版 | map/deque 查找与分配 | 两个固定环形历史、原 mutex 队列 | `O(N)` |
| 第三版 | 无界队列和逐条同步 | 单生产者、单消费者 | `O(N)`，同步按 batch 摊薄 |
| 第四版 | 单消费者 CPU 瓶颈 | 每个合约单写、有序 | 多 shard 并行 |
| 第五版 | 剩余同步/cache/NUMA | 前面已经验证的语义 | 依硬件与实现而定 |

## 8. 逐版 benchmark，而不是只报最后一个数字

原代码末尾的 `5.15096e+06 ticks/sec` 只是一次运行，不能直接作为可靠结论，因为它混合了：

- 每 Tick 一次 `high_resolution_clock::now()`；
- 生成数据的取模运算；
- 线程启动和调度；
- 队列传输与 SMA 计算；
- 未说明的编译选项、CPU 和系统负载。

建议为每一版填写同一张表：

| 版本 | kernel ticks/s | end-to-end ticks/s | P50 | P99 | max queue depth | checksum |
|---|---:|---:|---:|---:|---:|---:|
| 原始版 | 待测 | 待测 | 待测 | 待测 | 无界 | 待测 |
| 第一版 | 待测 | 待测 | 待测 | 待测 | 无界 | 必须一致 |
| 第二版 | 待测 | 待测 | 待测 | 待测 | 无界 | 必须一致 |
| 第三版 | 待测 | 待测 | 待测 | 待测 | 记录 | 必须一致 |
| 第四版 | 待测 | 待测 | 待测 | 待测 | 分 shard 记录 | 必须一致 |

性能指标之外，每一版还必须满足同一组功能指标：

```text
每个有效 Tick 恰好产生一条 SMA 输出
每个合约 price 不足 100 条时输出 NaN
第 100 条 price 开始输出有限 SMA
每个合约最终只保留最近 100 个 price 和最近 100 个 SMA
按逻辑时间顺序遍历环形历史时，两段数据保持对齐
```

### 8.1 基准测试规则

1. Release 编译：`-O3 -DNDEBUG -march=native -pthread`；
2. 使用固定输入回放，每一版处理完全相同的数据；
3. `steady_clock` 测耗时，交易所时间戳使用行情源字段；
4. 先 warm-up，再多次运行，报告中位数和波动范围；
5. 分开测试纯 SMA kernel、queue 和端到端；
6. 对每个合约逐点比较：warm-up 位置必须同为 NaN，成熟窗口必须在容差内一致；
7. 同时报吞吐和 P99，batch 提升吞吐不代表延迟更低；
8. 用 checksum 防止无观察者计算，但 checksum 不能替代逐点对拍；
9. 使用 `perf stat`、火焰图、ThreadSanitizer 和 UBSan 验证判断。

## 9. 贯穿所有版本的正确性问题

### 9.1 最近 100 笔不是最近 10 秒

原代码是 tick-count window。活跃合约 100 笔可能只覆盖几毫秒，冷门合约可能覆盖几分钟。如果需求是时间窗口，就要为每个合约保存 `(event_timestamp, price)`，并按事件时间淘汰。

此时还要定义：

- event time 还是 arrival time；
- 允许多大乱序；
- watermark 如何推进；
- 迟到数据丢弃、修正还是重算。

### 9.2 rolling sum 的数值误差

`double` 无法精确表示大多数十进制价格，长期执行 `sum += new-old` 会累积误差。可选方案：

- 定期完整重算窗口和；
- 用 `long double` 保存 sum；
- 有固定 tick size 时把价格转成整数 tick；
- 账务金额使用定点/decimal，不直接照搬指标计算的 `double`。

### 9.3 行情质量和过载

至少要监控或处理：

- sequence gap、重复包和乱序包；
- 非法 ID、NaN、Inf、异常成交量；
- 队列丢弃数量、最大深度和数据新鲜度；
- 交易暂停、复权、期货换月和合约生命周期；
- 生产者或消费者异常退出后的关闭和恢复。

## 10. 面试怎么回答

### 10.1 第一层：先说第一版

> 我不会一上来就把它改成 lock-free。先看原 consumer，每个成熟 Tick 都会对 price 和 SMA 历史执行 `vector.erase(begin())`，还要重新遍历 100 个 price 求和。第一版保留原来的 map、mutex、condition_variable 和单生产者单消费者，只把两段历史改成 deque，并为每个合约维护 rolling sum。每个 Tick 都向 SMA 历史写一条结果：price 不足 100 个就写 NaN，第 100 个开始写实际 SMA；两段历史都只保留最近 100 条。这样既保留原需求，又把窗口维护从 O(W) 降成 O(1)。

### 10.2 对方继续问，再说第二版

> 第一版之后还有每 Tick 的 map 查找和 deque 分段存储。模拟代码的 instrument_id 是连续的，所以第二版把合约状态改成 vector，下标直接定位；price window 和 SMA history 都固定为 100，就分别用 array 做环形缓冲区。SMA update 直接返回 double，warm-up 返回 NaN，并把这个返回值写入第二个环。这样既保持两段 100 条历史，也去掉 map 查找、节点分配和窗口动态分配，计算部分整体做到 O(N)。

### 10.3 对方继续问并发，再说第三、四版

> 计算优化后再 profile 队列。如果逐 Tick 锁和唤醒成为瓶颈，第三版先改成有界队列明确背压，再按数量或时间阈值批量传输，消费者把一批 Tick 移到本地后在锁外计算。需要多核时不能让多个消费者随便抢同一个队列，因为 SMA 有合约内顺序；第四版要按 instrument_id 稳定分片，让同一个合约始终由同一个 worker 更新。只有这些做完后同步仍是热点，我才会考虑每个 shard 使用 SPSC ring、绑核和 NUMA 优化。

## 11. 常见追问

### Q1：第一版为什么用 deque，不直接用 array？

因为第一版只想验证最明显的算法改动，并尽量少改原代码。`deque` 可以直接替换 vector 的头删，代码改动很小。固定 array 和环形下标属于第二版的数据布局优化，单独测量更容易说明收益来源。

### Q2：为什么第一版不删掉 map？

同样是为了单变量实验。第一版只证明 rolling sum 消除了 `O(W)`；第二版再证明连续状态消除了 `O(log M)` 和 pointer chasing。如果一起改，benchmark 只能看到总收益，解释不出每个改动的贡献。

### Q3：rolling sum 会不会算不准？

会积累浮点舍入误差，所以第二版加入低频完整重算进行校准。若价格有固定最小变动单位，还可以转成整数 tick 累加。

### Q4：为什么不能直接开四个 consumer？

因为同一合约的 SMA 必须按 Tick 顺序更新一个共享状态。直接抢队列可能产生数据竞争、乱序或把历史拆散。应按 instrument_id 分片，让一个合约只有一个写 worker。

### Q5：batch 越大越好吗？

不是。batch 越大，锁开销摊得越薄，但 Tick 等待凑批的时间越长。应设置数量和时间双阈值，并同时看吞吐和 P99。

### Q6：为什么最后才考虑 lock-free？

因为它只优化传输同步，不会解决原来的 `O(W)` 计算和 map 查找，而且内存序、关闭、满队列和忙等策略更容易写错。先完成低风险优化，再用 profiling 判断同步是否值得复杂化。

### Q7：为什么 warm-up 返回 NaN，不用 optional？

这里需要让 price 序列和 SMA 序列按 Tick 一一对齐，所以每个输入都必须占据一个输出位置。`NaN` 能明确表示“这个位置存在，但窗口尚未成熟”；使用 `optional` 也能表达未就绪，却会让调用方额外决定是否向历史写占位值。既然接口要求固定长度数值历史，就直接返回并保存 NaN，但后续聚合、比较和 checksum 必须显式用 `std::isnan()` 排除它。

## 12. 最终检查清单

- [ ] 第一版使用两个 deque 保留最近 100 个 price 和 SMA，并用 rolling sum；
- [ ] 每个有效 Tick 都产生 SMA 记录，前 99 个为 NaN，第 100 个开始有效；
- [ ] 第二版再把两段历史改为 fixed ring，并按逻辑时间顺序验证对齐；
- [ ] 第三版先定义背压，再增加 batch；
- [ ] 第四版按 instrument_id 分片，验证合约内 sequence 单调；
- [ ] 第五版只在 profiling 支持时进入；
- [ ] 明确是 Tick 数窗口还是时间窗口；
- [ ] 处理 NaN、Inf、非法 ID、重复、丢包和乱序；
- [ ] rolling sum 定期校准或使用定点价格；
- [ ] 每版 SMA 输出数、NaN 数、成熟结果数、保留历史数和 checksum 一致；
- [ ] 同时记录吞吐、P99、queue depth、CPU 和丢弃数；
- [ ] 使用固定回放、Release 编译和多轮统计；
- [ ] 使用 sanitizer 与 perf 验证正确性和瓶颈。
