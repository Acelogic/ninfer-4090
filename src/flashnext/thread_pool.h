// A small spin-barrier thread pool for latency-critical work: every call runs fn(thread_index) on all
// threads and returns when all have finished. Workers spin between calls, so a call costs a few
// microseconds rather than a wake-up from the OS scheduler.
#pragma once
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace ninfer::flashnext {

class SpinPool {
public:
    explicit SpinPool(int threads) : n_(threads < 1 ? 1 : threads) {
        for (int t = 1; t < n_; ++t) workers_.emplace_back([this, t] { loop(t); });
    }
    ~SpinPool() {
        stop_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto & w : workers_) w.join();
    }
    SpinPool(const SpinPool &) = delete;
    SpinPool & operator=(const SpinPool &) = delete;

    int size() const { return n_; }

    template <class F> void run(F && fn) {
        job_ = [&fn](int t) { fn(t); };
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        job_(0);
        while (done_.load(std::memory_order_acquire) != n_ - 1) _mm_pause();
    }

private:
    void loop(int t) {
        unsigned seen = 0;
        for (;;) {
            unsigned g;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) _mm_pause();
            seen = g;
            if (stop_.load(std::memory_order_acquire)) return;
            job_(t);
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::function<void(int)> job_;
    std::atomic<unsigned> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
};

}  // namespace ninfer::flashnext
