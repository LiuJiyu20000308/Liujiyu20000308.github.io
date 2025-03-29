#include <iostream>
#include <coroutine>
#include <string>

class CoroTask {
public:
  struct promise_type;
  using CoroHd = std::coroutine_handle<promise_type>;

  CoroTask(auto hd) : hd_{hd} {}
  ~CoroTask() {
    if (hd_) { hd_.destroy(); }
  }

  CoroTask(const CoroTask&) = delete;
  CoroTask& operator=(const CoroTask&) = delete;

  int getVlaue() const { return hd_.promise().CoroValue; }

  bool resume() {
    if (hd_ && !hd_.done()) {
        hd_.resume();
        return true;
    }
    return false;
  }

public:
  struct promise_type {
    int CoroValue{};
    /* data */
    auto get_return_object() {
        return CoroTask{CoroHd::from_promise(*this)};
    }
    auto initial_suspend() { return std::suspend_always{};}
    void unhandled_exception() { std::terminate();}
    auto yield_value(int val) {
      CoroValue = val;
      return std::suspend_always{};
    }
    void return_void() {}
    auto final_suspend() noexcept { return std::suspend_always{}; } 
  };

private:
  CoroHd hd_;
};

CoroTask coro(int max) {
    std::cout << "coro start, max: " << max << "\n";

    for (int i = 0; i <= max; ++i) {
        std::cout << "coro index: " << i << "\n";
        co_yield i;
    }
    std::cout << "coro end\n";
}

int main() {
    auto task = coro(4);
    while (task.resume()) {
        std::cout << "main get coroutine value: " << task.getVlaue() << "\n";
    }

    return 0;
}