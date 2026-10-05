#include "flashnext/prefill.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::flashnext {

StageProfile::~StageProfile() {
    for (cudaEvent_t e : pool_) cudaEventDestroy(e);
}

void StageProfile::begin(cudaStream_t s) {
    if (!on_) return;
    used_ = 0;
    names_.clear();
    mark(nullptr, s);
}

void StageProfile::mark(const char * stage, cudaStream_t s) {
    if (!on_) return;
    if (used_ == int(pool_.size())) {
        cudaEvent_t e = nullptr;
        if (cudaEventCreate(&e) != cudaSuccess) {
            on_ = false;
            return;
        }
        pool_.push_back(e);
    }
    cudaEventRecord(pool_[std::size_t(used_++)], s);
    names_.push_back(stage);
}

void StageProfile::end_chunk() {
    if (!on_ || used_ < 2) return;
    for (int i = 1; i < used_; ++i) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, pool_[std::size_t(i - 1)], pool_[std::size_t(i)]) != cudaSuccess) continue;
        const std::string n = names_[std::size_t(i)] ? names_[std::size_t(i)] : "?";
        if (!gpu_.count(n)) order_.push_back(n);
        gpu_[n] += ms;
    }
    used_ = 0;
    names_.clear();
    ++chunks_;
}

void StageProfile::report(std::FILE * f, const char * title) {
    if (!on_ || (gpu_.empty() && host_.empty())) return;
    double total = 0;
    for (const auto & [k, v] : gpu_) total += v;
    std::fprintf(f, "profile %s: %d chunks, %.1f ms on the GPU timeline\n", title, chunks_, total);
    for (const std::string & k : order_) std::fprintf(f, "  %-22s %9.1f ms  %5.1f%%\n", k.c_str(), gpu_[k], 100.0 * gpu_[k] / std::max(total, 1e-9));
    for (const auto & [k, v] : host_) std::fprintf(f, "  host %-17s %9.1f ms\n", k.c_str(), v);
    gpu_.clear();
    host_.clear();
    order_.clear();
    chunks_ = 0;
}

bool place_groups(std::vector<StreamGroup> & groups, std::size_t ring_bytes) {
    std::size_t at = 0;
    for (StreamGroup & g : groups) {
        if (g.bytes > ring_bytes) return false;
        if (at + g.bytes > ring_bytes) at = 0;
        g.ring_off = at;
        at += g.bytes;
    }
    return true;
}

std::vector<int> pick_cpu_share(const std::vector<int> & rest, const std::vector<double> & share, int T, double us_per_copy, const ShareModel & m) {
    std::vector<int> order(rest);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return share[std::size_t(a)] < share[std::size_t(b)]; });
    const double pairs = double(T) * 10.0;
    double cpu = 0;
    std::size_t k = 0;
    for (; k < order.size(); ++k) {
        const double c = m.us_per_expert + m.us_per_pair * pairs * share[std::size_t(order[k])];
        if (m.us_dense * T + cpu + c > us_per_copy * double(order.size() - k - 1)) break;
        cpu += c;
    }
    std::vector<int> out(order.begin(), order.begin() + std::ptrdiff_t(k));
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace ninfer::flashnext
