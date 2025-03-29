#include <iostream>
#include <vector>
#include <coroutine>
#include <exception>
#include <type_traits>

class Task
{
public:
    struct promise_type;
    using TaskHd1 = std::coroutine_handle<promise_type>;

private:
    TaskHd1  hd1_;

public:
    Task(auto h) : hd1_{h} {}
    ~Task() {
        if (hd1_) { hd1_.destroy(); }
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    bool resume() {
        if (!hd1_ || hd1_.done())
            return false;
        hd1_.resume();
        return true;
    }

public:
    struct promise_type {
        /* data */
        auto get_return_object() {
            return Task{TaskHd1::from_promise(*this)};
        }
        auto initial_suspend() { return std::suspend_always{};}
        void unhandled_exception() { std::terminate();}
        void return_void() {}
        auto final_suspend() noexcept { return std::suspend_always{}; }
    };
};

Task hello(int max) {
    std::cout << "hello world\n";
    for (int i = 0; i < max; ++i) {
        std::cout << "hello " << i << "\n";
        co_await std::suspend_always{};
    }

    std::cout << "hello end\n";
}

int main() {
    auto co = hello(3);
    while (co.resume()) {
        std::cout << "hello coroutine suspend\n";
    }
    return 0;
}