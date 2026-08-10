---
layout: post
title: C++记录：std::string
date: 2026-8-10 15:00 +0800
tags: [C++]
toc: true
---

## 面试问题：如何观察 `std::string` 什么时候分配内存？

可以先用下面这段话回答：

> `std::string` 对象本身通常位于栈上，但字符缓冲区不一定。较短的字符串通常使用 SSO，字符直接存放在 string 对象内部，不进行堆分配；长度超过实现的 SSO 容量后，才会申请堆内存。继续执行 `append`、`push_back` 或 `resize`，当新长度超过当前 `capacity()` 时，通常会重新分配更大的缓冲区。观察方法可以分两层：用 `size()`、`capacity()` 和 `data()` 地址变化间接观察扩容；如果要准确看到每一次申请，可以给 `std::basic_string` 传入自定义 allocator，在 `allocate()` 和 `deallocate()` 中打日志，也可以在 GDB 中对 `operator new` 或 `malloc` 下断点。

回答时要注意两个边界：

1. SSO 是标准库实现的优化，不是 C++ 标准强制要求，不能把某个固定长度说成所有平台都一样；
2. `capacity()` 或 `data()` 变化只能帮助观察缓冲区变化，自定义 allocator 或内存分析工具才能直接证明发生了动态分配。

## `std::string` 到底保存了什么

`std::string` 是下面这个类型的别名：

```cpp
using string = std::basic_string<char>;
```

它是一个拥有字符数据所有权的 RAII 对象。离开作用域时，string 会自动释放自己持有的动态内存。

```cpp
void example() {
    std::string text = "hello";
} // text 析构，相关资源自动释放
```

一个 string 对象通常需要记录：

- 当前字符串长度；
- 当前可用容量；
- 字符数据的位置，或者直接保存短字符串字符。

具体字段和布局由标准库实现决定，不能依赖某个实现的内部成员。

### string 对象和字符缓冲区不是一回事

```cpp
std::string text = makeText();
```

如果 `text` 是局部变量，string 对象本身通常位于当前栈帧；但长字符串的字符缓冲区通常由对象在堆上单独分配。string 对象内部保存指向该缓冲区的指针、长度和容量等信息。

所以 `sizeof(std::string)` 只表示 string 管理对象本身的固定大小，不是字符串长度，也不包含可能分配在堆上的全部字符空间：

```cpp
std::string a = "hi";
std::string b(100000, 'x');

std::cout << sizeof(a) << '\n';
std::cout << sizeof(b) << '\n';
// 二者相同，但 b 管理的字符数据远大于 a
```

### 连续存储和结尾的 `\0`

现代 C++ 保证 string 的字符连续存储，因此可以通过 `data()` 或 `c_str()` 与需要连续字符的接口交互：

```cpp
std::string text = "hello";

const char* p1 = text.data();
const char* p2 = text.c_str();

std::cout << p1 << '\n';
```

`c_str()` 返回以 `\0` 结尾的只读 C 字符串。C++17 起，非 const string 的 `data()` 返回可写的 `char*`，但只能修改 `[0, size())` 范围内已有字符，不能越过 size 写入，也不能破坏结尾字符。

## SSO：短字符串为什么可能不申请堆内存

SSO 是 Small String Optimization，即短字符串优化。string 对象内部通常预留一小块空间：

```text
短字符串：

string object
+-----------------------------------+
| size | inline characters ... | \0 |
+-----------------------------------+
             不需要堆分配

长字符串：

string object                    heap buffer
+----------------------+         +-----------------------+
| pointer | size | cap | ------> | characters ... | \0  |
+----------------------+         +-----------------------+
```

SSO 的优点是：

- 构造短字符串时避免 `new/malloc`；
- 减少内存碎片和 allocator 锁竞争；
- 字符与对象在同一片内存中，cache locality 更好。

SSO 能保存多少字符完全取决于实现、架构、字符类型和编译配置。例如某个 64 位 libstdc++ 实现可能观察到 15 个 `char` 的内联容量，但这不是可移植承诺。代码不能用固定阈值判断字符串是否在堆上。

## `size()`、`capacity()` 和内存分配

```cpp
std::string text = "hello";

std::cout << text.size() << '\n';
std::cout << text.capacity() << '\n';
```

- `size()`：当前实际字符数量，不包含结尾的 `\0`；
- `capacity()`：当前缓冲区在再次扩容前能够容纳的字符数量；
- `empty()`：等价于检查 `size() == 0`；
- `max_size()`：当前实现理论允许的最大长度，不代表机器当前真的能分配这么多内存。

通常有：

```text
size() <= capacity()
```

当追加后的长度不超过 capacity 时，可以复用当前缓冲区；超过 capacity 时，需要申请更大的缓冲区、移动或复制已有字符，再释放旧缓冲区。

### 为什么 capacity 往往不是每次只增加 1

如果每次 `push_back()` 都只申请刚好够一个新字符的空间，连续追加 $n$ 个字符会反复复制已有数据，总成本可能退化为 $O(n^2)$。标准库通常采用几何增长策略，一次多申请一些空间，使连续追加的均摊复杂度保持在 $O(1)$。

具体增长倍数由实现决定，不能依赖 capacity 一定翻倍或增加 1.5 倍。

### `reserve()` 为什么能减少分配

如果提前知道大致长度，可以先预留容量：

```cpp
std::string result;
result.reserve(1024);

for (int i = 0; i < 1000; ++i) {
    result.push_back('x');
}
```

没有 `reserve()` 时，追加过程中可能多次扩容；预留足够容量后，这些字符可以直接写入同一缓冲区。`reserve()` 可能立即触发一次分配，但它用一次可预测分配换掉了之后的多次扩容。

### `resize()` 和 `reserve()` 的区别

```cpp
std::string text = "abc";

text.reserve(100);  // size 仍为 3，只改变可用容量
text.resize(100);   // size 变为 100，新增字符进行初始化
```

`reserve()` 只准备存储空间，不产生新的逻辑字符；`resize()` 改变字符串长度，扩长时新增字符默认初始化为 `\0`，也可以指定填充值：

```cpp
text.resize(100, 'x');
```

### `clear()` 会不会释放内存

```cpp
text.clear();
```

`clear()` 将 size 变为 0，但通常保留 capacity，目的是让之后的写入复用缓冲区。不能用 `clear()` 保证归还堆内存。

如果希望请求缩小容量，可以使用：

```cpp
text.shrink_to_fit();
```

但 `shrink_to_fit()` 只是非强制请求，实现可以选择不缩容；即使执行缩容，它本身也可能先分配新的较小缓冲区，再迁移数据。

## 方法一：观察 capacity 和 data 地址

下面的程序能看到每次长度变化后，capacity 和字符地址是否改变：

```cpp
#include <iostream>
#include <string>

int main() {
    std::string text;

    std::cout << "size\tcapacity\tdata\n";
    for (int i = 0; i < 100; ++i) {
        const char* oldData = text.data();
        std::size_t oldCapacity = text.capacity();

        text.push_back('a');

        if (text.capacity() != oldCapacity || text.data() != oldData) {
            std::cout << text.size() << '\t'
                      << text.capacity() << '\t'
                      << static_cast<const void*>(text.data()) << '\n';
        }
    }
}
```

如果 capacity 增加且 `data()` 地址改变，说明字符存储发生了迁移，通常对应一次重新分配。第一次从对象内部的 SSO 缓冲区切换到堆缓冲区时，也能观察到地址变化。

这种方法简单，但属于间接观察：

- 不能直接显示申请了多少字节；
- 不能区分标准库内部的所有分配细节；
- 不同实现的增长策略和 SSO 阈值不同，输出也不同。

## 方法二：使用自定义 allocator 精确记录

`std::string` 实际是 `std::basic_string`，它允许传入 allocator。下面的 allocator 会在每次动态申请和释放时打印字节数与地址：

```cpp
#include <cstdio>
#include <new>
#include <string>
#include <string_view>

template <class T>
class LoggingAllocator {
public:
    using value_type = T;

    LoggingAllocator() noexcept = default;

    template <class U>
    LoggingAllocator(const LoggingAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        std::size_t bytes = count * sizeof(T);
        void* memory = ::operator new(bytes);
        std::fprintf(stderr, "allocate   %zu bytes at %p\n", bytes, memory);
        return static_cast<T*>(memory);
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        std::fprintf(stderr, "deallocate %zu bytes at %p\n",
                     count * sizeof(T), static_cast<void*>(pointer));
        ::operator delete(pointer);
    }
};

template <class T, class U>
bool operator==(const LoggingAllocator<T>&,
                const LoggingAllocator<U>&) noexcept {
    return true;
}

template <class T, class U>
bool operator!=(const LoggingAllocator<T>& lhs,
                const LoggingAllocator<U>& rhs) noexcept {
    return !(lhs == rhs);
}

using LoggingString = std::basic_string<
    char,
    std::char_traits<char>,
    LoggingAllocator<char>>;

void printState(std::string_view operation, const LoggingString& text) {
    std::fprintf(stderr,
                 "%-16.*s size=%zu capacity=%zu data=%p\n",
                 static_cast<int>(operation.size()), operation.data(),
                 text.size(), text.capacity(),
                 static_cast<const void*>(text.data()));
}

int main() {
    LoggingString text;
    printState("constructed", text);

    text.append("short");
    printState("append short", text);

    text.append(100, 'x');
    printState("append 100", text);

    text.reserve(256);
    printState("reserve 256", text);

    text.clear();
    printState("clear", text);

    text.shrink_to_fit();
    printState("shrink_to_fit", text);
}
```

典型现象是：

1. 构造空字符串时没有调用 allocator；
2. `append short` 仍位于 SSO 缓冲区，没有堆分配；
3. 追加 100 个字符超过 SSO 容量，出现 `allocate`；
4. `reserve(256)` 超过当前 capacity，再次分配并释放旧缓冲区；
5. `clear()` 只把 size 设为 0，不释放原缓冲区；
6. `shrink_to_fit()` 可能释放堆缓冲区，并让空字符串重新回到 SSO 状态。

实际输出取决于标准库实现。特别是 SSO 容量、增长策略和 `shrink_to_fit()` 是否执行都不能写死。

### 为什么 allocator 方法比重载全局 new 更适合

重载全局 `operator new` 或在 `malloc` 上打点会看到整个进程的所有分配，包括 iostream、容器和运行库产生的噪声。自定义 allocator 只记录这个 `basic_string` 的字符存储申请，因果关系更清楚。

但它也有边界：它观察的是使用该 allocator 的实验 string，不会自动拦截程序中所有普通 `std::string`。如果需要分析现有大型程序，调试器或内存 profiler 更合适。

## 方法三：GDB 和内存分析工具

### GDB 断点

调试小程序时可以尝试：

```text
(gdb) break operator new(unsigned long)
(gdb) break malloc
(gdb) run
(gdb) backtrace
```

命中后通过调用栈判断分配是否来自 `std::string`。实际符号名可能受编译器、优化级别和动态链接影响；`malloc` 断点还会捕获其他库分配，因此最好使用最小复现程序，并关闭或降低优化方便观察。

### Heap profiler

分析真实程序时可以使用：

- Heaptrack：查看分配次数、字节数和调用栈；
- Valgrind Massif：分析堆内存随时间的变化；
- 自定义 malloc/new hook 或采样分配 profiler：用于大型服务中的低开销观测。

这类工具适合回答“哪个调用路径分配最多”，而 `capacity()` 更适合解释某一个 string 为什么在某一步扩容。

## 哪些操作可能触发分配

| 操作 | 是否可能分配 | 原因 |
|---|---:|---|
| 构造短字符串 | 不一定 | 可能使用 SSO |
| 构造长字符串 | 是 | 字符超过对象内联容量 |
| `push_back/append/insert/operator+=` | 是 | 新 size 超过 capacity 时扩容 |
| `replace/resize` 扩长 | 是 | 需要的字符空间超过 capacity |
| `reserve(n)` | 是 | `n > capacity()` 时需要更大缓冲区 |
| `clear()` | 通常否 | 只改变 size，保留容量供复用 |
| `erase()` 缩短 | 通常否 | 删除字符不要求自动缩容 |
| `shrink_to_fit()` | 可能 | 非强制缩容请求，可能迁移缓冲区 |
| 拷贝长字符串 | 通常是 | 目标 string 需要拥有独立字符数据 |
| 移动字符串 | 不一定 | 长字符串可能转移缓冲区，SSO 字符仍需复制 |
| `substr()` | 可能 | 返回一个拥有独立数据的新 string |
| 多次 `operator+` | 可能多次 | 中间临时 string 可能分配和扩容 |
| `find/compare/读取字符` | 否 | 只读取现有数据 |

“可能”是因为标准只规定行为和复杂度边界，具体 SSO、容量复用和增长策略属于实现细节。

## 常用构造与访问

```cpp
std::string a;                    // 空字符串
std::string b = "hello";          // 从 C 字符串构造
std::string c(5, 'x');            // "xxxxx"
std::string d(b, 1, 3);           // "ell"
std::string e(b.begin(), b.end());
```

字符访问：

```cpp
char first = b.front();
char last  = b.back();
char x     = b[1];
char y     = b.at(1);
```

`operator[]` 不做常规越界检查；`at()` 越界时抛出 `std::out_of_range`。在性能敏感且边界已经由逻辑保证时常用 `[]`，处理外部输入或需要明确错误时可以使用 `at()`。

## 修改字符串

```cpp
std::string text = "abc";

text.push_back('d');          // "abcd"
text += "ef";                // "abcdef"
text.append("gh");           // "abcdefgh"
text.insert(2, "XX");        // "abXXcdefgh"
text.erase(2, 2);             // 删除两个字符
text.replace(0, 2, "AB");    // 替换区间
text.pop_back();              // 删除末尾字符
text.clear();                 // size 变为 0
```

在中间执行 `insert()` 或 `erase()` 通常需要移动后面的字符，时间复杂度为 $O(n)$；尾部追加在不扩容时更便宜。

## 查找、截取和比较

```cpp
std::string text = "database engine";

std::size_t pos = text.find("engine");
if (pos != std::string::npos) {
    std::string part = text.substr(pos, 6);
}

if (text.starts_with("data")) { // C++20
}

if (text.ends_with("engine")) { // C++20
}
```

`find()` 找不到时返回 `std::string::npos`，它是 `size_type` 的最大值，不能用 `-1` 风格的有符号整数逻辑随意混用。

字符串比较按字典序进行：

```cpp
if (a == b) {}
if (a < b)  {}

int result = a.compare(b);
// result < 0、== 0、> 0
```

## `std::string`、C 字符串和 `string_view`

### `std::string`

- 拥有字符数据；
- 可以修改长度；
- 负责生命周期；
- 构造、拷贝或扩容时可能分配。

### `const char*`

- 只是指针，不表达长度和所有权；
- 常以 `\0` 表示结尾；
- 指针可能悬空；
- 不能直接表示中间包含 `\0` 的完整二进制数据长度。

### `std::string_view`

- 只保存指针和长度，不拥有数据；
- 构造和传参通常不分配；
- 适合只读参数和切片；
- 原数据销毁、移动或重新分配后，view 会悬空。

```cpp
std::string_view prefix(std::string_view text, std::size_t n) {
    return text.substr(0, std::min(n, text.size()));
}
```

下面的代码是错误的：

```cpp
std::string_view badView() {
    std::string local = "temporary";
    return local; // local 析构后，返回的 view 悬空
}
```

## 指针、引用和迭代器什么时候失效

```cpp
std::string text = "hello";
const char* pointer = text.data();

text += std::string(1000, 'x');
// 扩容后 pointer 很可能已经悬空
```

任何可能改变 string 存储的操作之后，都不应继续盲目使用之前取得的 `data()`、`c_str()`、引用或迭代器。安全做法是在修改完成后重新获取。

即使本次测试发现地址没变，也不能把它推广为所有输入和实现都不会失效，因为下一次操作可能刚好越过 capacity。

## 拷贝、移动和返回值

```cpp
std::string a = "a long string ...";
std::string b = a;            // 拷贝，b 拥有独立内容
std::string c = std::move(a); // 移动，a 仍有效但值未指定
```

长字符串的移动通常可以转移堆缓冲区，避免复制全部字符；SSO 字符位于对象内部，移动时仍可能复制少量内联字符。自定义 allocator 不兼容时，移动赋值也可能不得不重新分配。因此不要无条件声称 string move 永远是 $O(1)$ 或永远不分配。

现代标准库实现通常不采用 Copy-On-Write。拷贝后修改一个 string 不会改变另一个 string：

```cpp
std::string x = "abc";
std::string y = x;
y[0] = 'X';

// x == "abc", y == "Xbc"
```

函数按值返回 string 通常可以依靠返回值优化或移动语义：

```cpp
std::string makeName() {
    std::string result = "Jiyu";
    return result; // 不要写 std::move(result)，避免阻碍 NRVO
}
```

## 容易忽略的正确性问题

### 字符串可以包含 `\0`

```cpp
std::string a("A\0B");     // 长度为 1，按 C 字符串构造
std::string b("A\0B", 3);  // 长度为 3，包含中间的 \0
```

`std::cout << b` 会按 string 长度输出全部字符，但传给只认 C 字符串的接口后，处理中间 `\0` 的行为取决于该接口，常常会在第一个 `\0` 截断。

### `getline()` 和 `operator>>`

```cpp
std::string word;
std::cin >> word;        // 遇到空白停止

std::string line;
std::getline(std::cin, line); // 读取整行
```

在 `operator>>` 后立即调用 `getline()` 时，输入流中可能残留换行符，可以先使用 `std::getline(std::cin >> std::ws, line)`，但 `std::ws` 也会跳过开头的所有空白，要根据业务语义选择。

### `tolower()` 和有符号 char

```cpp
char lower = static_cast<char>(
    std::tolower(static_cast<unsigned char>(ch)));
```

除 EOF 外，将负的 `char` 直接传给 `<cctype>` 函数可能产生未定义行为，因此应先转换为 `unsigned char`。

## 性能优化原则

### 已知输出规模时先 reserve

```cpp
std::string join(const std::vector<std::string>& parts) {
    std::size_t total = 0;
    for (const auto& part : parts) {
        total += part.size();
    }

    std::string result;
    result.reserve(total);
    for (const auto& part : parts) {
        result += part;
    }
    return result;
}
```

### 避免在循环中制造不必要的临时字符串

```cpp
// 可能反复产生临时对象和分配
result = result + prefix + value;

// 更直接
result.reserve(result.size() + prefix.size() + value.size());
result.append(prefix);
result.append(value);
```

### 只读参数优先考虑引用或 string_view

```cpp
void storeOwned(std::string text);          // 需要获得所有权
void readString(const std::string& text);   // 只接受 string，避免拷贝
void parse(std::string_view text);          // 接受多种连续字符来源
```

不要为了“零拷贝”到处保存 string_view。只有能够证明底层字符生命周期覆盖 view 使用期时才安全。

### 优化前实际测量

SSO、增长策略、allocator 和调用方式会共同影响结果。性能敏感代码应使用真实 workload，记录：

- 分配次数和总字节数；
- 平均及尾延迟；
- append 前后的 capacity；
- 临时 string 数量；
- cache miss 和数据复制量。

不要仅根据字符串长度猜测性能。

## 面试高频问答

### `clear()` 后内存一定释放吗？

不一定。`clear()` 只保证 size 变为 0，通常保留 capacity 供以后复用。`shrink_to_fit()` 也只是缩容请求，不保证一定释放。

### `reserve()` 和 `resize()` 的区别？

`reserve()` 改变容量但不产生逻辑字符；`resize()` 改变 size，扩长时会构造新增字符。

### 为什么 string append 的均摊复杂度可以是 O(1)？

标准库通常进行几何扩容，不是每次只增加一个字符。虽然某次扩容需要 $O(n)$ 复制，但扩容次数是对数级，分摊到多次尾部追加后为均摊 $O(1)$。

### 怎么判断当前 string 是否正在使用 SSO？

标准没有提供 `is_sso()` 接口。可以通过自定义 allocator 观察是否发生堆申请，或者在特定实现的调试实验中比较 `data()` 与对象地址，但不能把实验阈值写成可移植业务逻辑。

### `data()` 返回的指针能保存多久？

只应在 string 未执行可能改变存储的操作期间使用。append、insert、replace、resize、reserve 等操作可能重新分配并使旧指针失效，修改后应重新获取。

### 为什么 `std::string` 不能简单看成 `char*`？

string 同时管理所有权、长度、容量、结尾字符和异常安全；`char*` 只是地址，不知道缓冲区长度、容量及由谁释放。

### 如何减少 string 的内存分配？

先通过 profiler 确认分配热点，然后根据可预测长度使用 `reserve()`，用 `append()` 代替产生多层临时对象的拼接，只读接口使用引用或 `string_view`，并避免在热循环中反复构造和销毁大字符串。

## 这道面试题的完整回答模板

> 我会先区分 string 对象和字符缓冲区。短字符串在常见标准库中会使用 SSO，字符直接放在对象内部，所以构造时可能没有堆分配；超过 SSO 容量后才会申请外部缓冲区。后续 append 或 resize 超过当前 capacity 时会触发扩容，原来的 data 指针可能失效。简单观察可以在每次操作前后打印 size、capacity 和 data 地址；如果需要准确证明，我会把 string 替换成带 LoggingAllocator 的 basic_string，在 allocate/deallocate 中记录大小、地址和调用时机。分析现有程序则可以在 GDB 对 operator new/malloc 下断点，或使用 heaptrack 查看调用栈。SSO 阈值和扩容倍数属于实现细节，不能写死。
