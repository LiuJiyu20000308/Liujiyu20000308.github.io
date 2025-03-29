#include <iostream>
#include <vector>
#include <coroutine>
#include <string>
#include <numeric>


template <typename T>
class CoGen {
public:
  class promise_type {
  public:
    CoGen get_return_object() {
      return CoGen{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    void return_value(const std::vector<T>& vec) {
      vec_ = vec;
    }
    auto initial_suspend() { return std::suspend_always{};}
    void unhandled_exception() { std::terminate();}
    auto final_suspend() noexcept { return std::suspend_always{}; } 
    const std::vector<T>& getResult() const { return vec_; }

  private:
    std::vector<T> vec_{};
  };
  CoGen(auto hd) : hd_{hd} {}
  ~CoGen() {
    if (hd_) {
       hd_.destroy();
    }
  }

  std::vector<T> getResult() const { return hd_.promise().getResult(); }
  bool resume() {
    if (!hd_ || hd_.done()) return false;
    hd_.resume();
    return true;
  }

private:
  std::coroutine_handle<promise_type> hd_{};
};

CoGen<int> coroGen(int max) {
    std::vector<int> vec;
    vec.resize(abs(max));
    std::iota(vec.begin(), vec.end(), 0);
    co_return vec;
}

int main() {
    auto task = coroGen(7);
    while (task.resume());
    std::cout << "coroutine end\n";
    for (const auto& val : task.getResult()) {
        std::cout << " " << val;
    }
    std::cout << "\n";
    return 0;
}
