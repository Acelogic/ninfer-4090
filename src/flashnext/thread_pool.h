// A small spin-barrier thread pool for latency-critical work: every call runs fn(thread_index) on all
// threads and returns when all have finished. Workers spin between calls, so a call costs a few
// microseconds rather than a wake-up from the OS scheduler. A worker that has seen no call for
// kIdleSpin goes to sleep on the generation counter, so an idle pool (a server between requests)
// costs no CPU; decoding calls the pool every few hundred microseconds and never lets it sleep.
#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace ninfer::flashnext {

class SpinPool {
public:
    static constexpr std::chrono::microseconds kIdleSpin{2000};

    explicit SpinPool(int threads) : n_(threads < 1 ? 1 : threads) {
        for (int t = 1; t < n_; ++t) workers_.emplace_back([this, t] { loop(t); });
    }
    ~SpinPool() {
        stop_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_seq_cst);
        gen_.notify_all();
        for (auto & w : workers_) w.join();
    }
    SpinPool(const SpinPool &) = delete;
    SpinPool & operator=(const SpinPool &) = delete;

    int size() const { return n_; }

    template <class F> void run(F && fn) {
        job_ = [&fn](int t) { fn(t); };
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_seq_cst);
        // a sleeping worker registers before it checks the counter, so it is either seen here or sees
        // the new value (both are sequentially consistent)
        if (sleepers_.load(std::memory_order_seq_cst) > 0) gen_.notify_all();
        job_(0);
        while (done_.load(std::memory_order_acquire) != n_ - 1) _mm_pause();
    }

private:
    void loop(int t) {
        unsigned seen = 0;
        for (;;) {
            unsigned g = gen_.load(std::memory_order_acquire);
            if (g == seen) {
                const auto t0 = std::chrono::steady_clock::now();
                for (unsigned spin = 1; (g = gen_.load(std::memory_order_acquire)) == seen; ++spin) {
                    _mm_pause();
                    if ((spin & 1023) == 0 && std::chrono::steady_clock::now() - t0 > kIdleSpin) {
                        sleepers_.fetch_add(1, std::memory_order_seq_cst);
                        while ((g = gen_.load(std::memory_order_seq_cst)) == seen) gen_.wait(seen, std::memory_order_seq_cst);
                        sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                        break;
                    }
                }
            }
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
    std::atomic<int> sleepers_{0};
    std::atomic<bool> stop_{false};
};

}  // namespace ninfer::flashnext
