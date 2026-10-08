#include "flashnext/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "flashnext/cuda/ops.h"
#include "flashnext/cuda/qsa.h"

namespace ninfer::flashnext {

namespace fc = cuda;
using fc::check;

namespace {
constexpr std::size_t kRowBytes = std::size_t(fc::kKvHeads) * fc::kHeadDim * sizeof(half);  // one position of K (or V)
}

struct KvStreamCache::Layer {
    half * host_k = nullptr;  // pinned host copies (host pointers) and their device-mapped addresses
    half * host_v = nullptr;
    half * dev_k = nullptr;
    half * dev_v = nullptr;
    fc::DeviceBuffer pool_k, pool_v, map_mem;
    fc::KvStreamMap map;
    ~Layer() {
        if (host_k) cudaFreeHost(host_k);
        if (host_v) cudaFreeHost(host_v);
    }
};

KvStreamCache::KvStreamCache(std::int64_t max_ctx, std::int64_t resident_cells, int group_tokens, cudaStream_t stream)
    : max_ctx_(max_ctx), resident_(resident_cells), n_slots_(resident_cells / fc::kKvPage), group_tokens_(group_tokens), stream_(stream) {
    if (resident_ < kMinResident || resident_ % fc::kKvPage != 0)
        throw std::runtime_error("engine: the KV page cache needs a multiple of 4 cells, at least " + std::to_string(kMinResident));
    // a query names at most 513 selected blocks and its own incomplete one
    group_ = int(n_slots_ / (fc::kQsaWidth / fc::kKvPage + 2));
    check(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "kv stream");
    check(cudaEventCreateWithFlags(&ev_free_, cudaEventDisableTiming), "kv event");
    check(cudaEventCreateWithFlags(&ev_staged_, cudaEventDisableTiming), "kv event");
}

KvStreamCache::~KvStreamCache() {
    if (copy_) cudaStreamSynchronize(copy_);
    layers_.clear();
    if (ev_free_) cudaEventDestroy(ev_free_);
    if (ev_staged_) cudaEventDestroy(ev_staged_);
    if (copy_) cudaStreamDestroy(copy_);
}

int KvStreamCache::add_layer() {
    auto L = std::make_unique<Layer>();
    const std::size_t host_bytes = std::size_t(max_ctx_) * kRowBytes;
    for (half ** h : {&L->host_k, &L->host_v})
        check(cudaHostAlloc(reinterpret_cast<void **>(h), host_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "KV host copy");
    check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&L->dev_k), L->host_k, 0), "KV host copy");
    check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&L->dev_v), L->host_v, 0), "KV host copy");
    L->pool_k = fc::DeviceBuffer(std::size_t(resident_) * kRowBytes);
    L->pool_v = fc::DeviceBuffer(std::size_t(resident_) * kRowBytes);
    const std::int64_t n_blocks = (max_ctx_ + fc::kKvPage - 1) / fc::kKvPage + 1;
    L->map_mem = fc::DeviceBuffer(fc::kv_map_bytes(n_blocks, n_slots_));
    L->map = fc::kv_map_layout(L->map_mem.get(), n_blocks, n_slots_);
    fc::kv_map_reset(L->map, stream_);
    check(cudaStreamSynchronize(stream_), "KV map");
    layers_.push_back(std::move(L));
    return int(layers_.size()) - 1;
}

std::size_t KvStreamCache::stage_bytes(std::int64_t pos_end) const {
    if (pos_end <= resident_) return 0;
    return 2 * std::size_t((pos_end + 63) / 64 * 64) * kRowBytes;
}

void KvStreamCache::begin_step(std::int64_t p0, int T, void * stage) {
    p0_ = p0;
    T_ = T;
    staged_ = -1;
    if (T <= fc::kMaxTokens) {
        mode_ = Mode::Decode;
    } else if (p0 + T <= resident_) {
        mode_ = Mode::Prefix;
    } else if (group_tokens_ > 0 && T <= std::max<std::int64_t>(group_tokens_, p0 / kGroupDepth)) {
        mode_ = Mode::Groups;
        ++grouped_chunks_;
    } else {
        mode_ = Mode::Stage;
        if (!stage) throw std::runtime_error("engine: no staging pool for a prompt chunk beyond the KV page cache");
        const std::size_t rows = std::size_t((p0 + T + 63) / 64 * 64);
        stage_k_ = static_cast<half *>(stage);
        stage_v_ = stage_k_ + rows * fc::kKvHeads * fc::kHeadDim;
        // the first layer's rows start moving now, beside the layers before it
        check(cudaEventRecord(ev_free_, stream_), "kv stage");
        if (!layers_.empty()) issue_stage(0);
        ++staged_chunks_;
    }
}

void KvStreamCache::issue_stage(int li) {
    // the copy stream waits until the attention before has read the pool (ev_free), then DMAs [0, p0)
    const Layer & L = *layers_[std::size_t(li)];
    check(cudaStreamWaitEvent(copy_, ev_free_, 0), "kv stage");
    const std::size_t bytes = std::size_t(p0_) * kRowBytes;
    if (bytes) {
        check(cudaMemcpyAsync(stage_k_, L.host_k, bytes, cudaMemcpyHostToDevice, copy_), "kv stage");
        check(cudaMemcpyAsync(stage_v_, L.host_v, bytes, cudaMemcpyHostToDevice, copy_), "kv stage");
        staged_bytes_ += 2.0 * double(bytes);
    }
    check(cudaEventRecord(ev_staged_, copy_), "kv stage");
    cudaStreamQuery(copy_);  // submit now (Windows batches work otherwise)
    staged_ = li;
}

fc::KvStore KvStreamCache::store(int li) const {
    const Layer & L = *layers_[std::size_t(li)];
    fc::KvStore st;
    if (mode_ == Mode::Stage) {  // stage (read by this chunk's attention), host copy, resident pages
        st.k = stage_k_;
        st.v = stage_v_;
        st.k2 = L.dev_k;
        st.v2 = L.dev_v;
    } else {
        st.k = L.dev_k;
        st.v = L.dev_v;
    }
    st.page_table = L.map.page_table;
    st.kp = L.pool_k.as<half>();
    st.vp = L.pool_v.as<half>();
    return st;
}

void KvStreamCache::attend(int li, const float * q, const float * gate, std::int32_t * cells, const std::int32_t * n_cells,
                           const std::int64_t * d_pos0, float scale, float * work, float * out) {
    Layer & L = *layers_[std::size_t(li)];
    const int W = fc::kQsaWidth;
    half * pk = L.pool_k.as<half>();
    half * pv = L.pool_v.as<half>();
    switch (mode_) {
    case Mode::Decode:
        fc::kv_resolve(L.map, cells, n_cells, T_, W, true, stream_);
        fc::kv_copy_misses(L.map, L.dev_k, L.dev_v, pk, pv, stream_);
        fc::attn_sparse(q, gate, pk, pv, cells, n_cells, T_, scale, work, out, stream_);
        break;
    case Mode::Prefix:
        fc::kv_resolve_prefix(L.map, d_pos0, T_, stream_);
        fc::kv_copy_misses(L.map, L.dev_k, L.dev_v, pk, pv, stream_);
        fc::kv_translate(L.map, cells, n_cells, T_, W, stream_);
        fc::attn_sparse(q, gate, pk, pv, cells, n_cells, T_, scale, work, out, stream_);
        break;
    case Mode::Groups:
        // each group is resolved like a decode step and attends with the whole chunk's split count, so every
        // token rounds as in one call over all T
        for (int t0 = 0; t0 < T_; t0 += group_) {
            const int n = std::min(group_, T_ - t0);
            const std::size_t qo = std::size_t(t0) * fc::kHeads * fc::kHeadDim;
            fc::kv_resolve(L.map, cells + std::size_t(t0) * W, n_cells + t0, n, W, true, stream_);
            fc::kv_copy_misses(L.map, L.dev_k, L.dev_v, pk, pv, stream_);
            fc::attn_sparse(q + qo, gate + qo, pk, pv, cells + std::size_t(t0) * W, n_cells + t0, n, scale, work, out + qo, stream_, T_);
        }
        break;
    case Mode::Stage:
        if (staged_ != li) issue_stage(li);
        check(cudaStreamWaitEvent(stream_, ev_staged_, 0), "kv stage");
        fc::attn_sparse(q, gate, stage_k_, stage_v_, cells, n_cells, T_, scale, work, out, stream_);
        // the pool is free once this attention has run: start moving the next layer's rows
        check(cudaEventRecord(ev_free_, stream_), "kv stage");
        if (li + 1 < int(layers_.size())) issue_stage(li + 1);
        break;
    }
}

KvStreamStats KvStreamCache::stats() const {
    KvStreamStats s;
    s.enabled = true;
    s.resident_cells = resident_;
    s.layers = std::int64_t(layers_.size());
    double vram = 0, host = 0;
    for (const auto & L : layers_) {
        vram += double(L->pool_k.bytes() + L->pool_v.bytes() + L->map_mem.bytes());
        host += 2.0 * double(max_ctx_) * double(kRowBytes);
        const fc::KvStreamCounters c = fc::kv_counters(L->map);
        s.misses += c.misses;
        s.lookups += c.lookups;
        s.resolves += c.calls;
        s.overflow = s.overflow || c.overflow;
    }
    s.vram_gib = vram / double(1 << 30);
    s.host_gib = host / double(1 << 30);
    s.staged_chunks = staged_chunks_;
    s.grouped_chunks = grouped_chunks_;
    s.staged_gib = staged_bytes_ / double(1 << 30);
    return s;
}

std::size_t KvStreamCache::row_bytes() { return kRowBytes; }

void KvStreamCache::read_rows(int li, std::int64_t end, std::uint8_t * k, std::uint8_t * v) const {
    if (end < 0 || end > max_ctx_) throw std::runtime_error("engine: KV rows out of range");
    const Layer & L = *layers_[std::size_t(li)];
    std::memcpy(k, L.host_k, std::size_t(end) * kRowBytes);
    std::memcpy(v, L.host_v, std::size_t(end) * kRowBytes);
}

void KvStreamCache::write_rows(int li, std::int64_t end, const std::uint8_t * k, const std::uint8_t * v) {
    if (end < 0 || end > max_ctx_) throw std::runtime_error("engine: KV rows out of range");
    Layer & L = *layers_[std::size_t(li)];
    std::memcpy(L.host_k, k, std::size_t(end) * kRowBytes);
    std::memcpy(L.host_v, v, std::size_t(end) * kRowBytes);
}

void KvStreamCache::reset_pages() {
    for (auto & L : layers_) fc::kv_map_reset(L->map, stream_);
    check(cudaStreamSynchronize(stream_), "KV map");
}

}  // namespace ninfer::flashnext
