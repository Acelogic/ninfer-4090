#include "flashnext/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <thread>
#include <utility>

#include <cuda_runtime.h>
#include <immintrin.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "flashnext/cpu_experts.h"
#include "flashnext/cuda/experts.h"
#include "flashnext/cuda/experts_batch.h"
#include "flashnext/cuda/gemm.h"
#include "flashnext/cuda/gemv.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/cuda/qsa.h"
#include "flashnext/kv_cache.h"
#include "flashnext/quants.h"
#include "flashnext/reference.h"

namespace ninfer::flashnext {

namespace fc = cuda;
using fc::check;
using clk = std::chrono::steady_clock;

namespace {

fc::DeviceBuffer upload_f32(const GgufTensor & t, std::int64_t want) {
    if (t.elements() != want) throw std::runtime_error("engine: " + t.name + " has " + std::to_string(t.elements()) + " elements");
    std::vector<float> v(std::size_t(t.elements()));
    dequantize_row(t.type, t.data, v.data(), t.elements());
    fc::DeviceBuffer b(v.size() * sizeof(float));
    check(cudaMemcpy(b.get(), v.data(), b.bytes(), cudaMemcpyHostToDevice), "upload");
    return b;
}

fc::DeviceBuffer zeros(std::size_t bytes) {
    fc::DeviceBuffer b(bytes);
    check(cudaMemset(b.get(), 0, bytes), "memset");
    return b;
}

template <class T>
class Pinned {
public:
    explicit Pinned(std::size_t n) { check(cudaMallocHost(reinterpret_cast<void **>(&p_), n * sizeof(T)), "cudaMallocHost"); }
    ~Pinned() { cudaFreeHost(p_); }
    Pinned(const Pinned &) = delete;
    Pinned & operator=(const Pinned &) = delete;
    T * get() const { return p_; }

private:
    T * p_ = nullptr;
};

std::uint64_t free_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
#else
    return std::uint64_t(sysconf(_SC_AVPHYS_PAGES)) * std::uint64_t(sysconf(_SC_PAGESIZE));
#endif
}

// Drops part of the memory-mapped GGUF from the process's working set. The pages stay in the OS
// file cache as standby memory (available to anyone) and are read back if touched again.
void release_mapped(const void * p, std::size_t n) {
#ifdef _WIN32
    VirtualUnlock(const_cast<void *>(p), n);  // on pages that are not locked, this only trims them
#else
    const std::uintptr_t page = std::uintptr_t(sysconf(_SC_PAGESIZE));
    const std::uintptr_t a = (std::uintptr_t(p) + page - 1) & ~(page - 1), b = (std::uintptr_t(p) + n) & ~(page - 1);
    if (b > a) madvise(reinterpret_cast<void *>(a), b - a, MADV_DONTNEED);
#endif
}

// Asks the OS to read part of the memory-mapped GGUF into its file cache in the background.
void prefetch_mapped(const void * p, std::size_t n) {
#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY range{const_cast<void *>(p), n};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
#else
    const std::uintptr_t page = std::uintptr_t(sysconf(_SC_PAGESIZE));
    const std::uintptr_t a = std::uintptr_t(p) & ~(page - 1);
    madvise(reinterpret_cast<void *>(a), std::uintptr_t(p) + n - a, MADV_WILLNEED);
#endif
}

void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(std::string("engine: model does not match the compiled shapes: ") + what);
}

}  // namespace

struct Engine::Impl {
    const GgufModel & model;
    ReferenceConfig cfg;
    EngineOptions opt;
    EngineHook hook;
    EngineStats stats;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev_router = nullptr;
    std::unique_ptr<CpuExperts> experts;

    struct Layer {
        bool recurrent = false, ple = false;
        fc::DeviceWeight hc_attn_down, hc_attn_up, hc_attn_inject, hc_ffn_down, hc_ffn_up, hc_ffn_inject;
        fc::DeviceBuffer hc_attn_norm, hc_ffn_norm;
        fc::DeviceWeight wq, wk, wv, wo, idx_k, idx_q;  // attention, QSA indexer
        fc::DeviceBuffer q_norm, k_norm, idx_q_norm, idx_k_norm;
        fc::DeviceWeight wqkv, wgate, ssm_beta, ssm_alpha, ssm_out;  // Gated DeltaNet
        fc::DeviceBuffer conv1d, dt, a, ssm_norm;
        fc::DeviceWeight ple_key, ple_value;  // PLE
        fc::DeviceBuffer ple_nk, ple_nq, ple_nc, ple_conv;
        fc::DeviceWeight router, sh_gate, sh_up, sh_down, sh_gate_inp;  // FFN
        fc::DeviceBuffer conv_state, S, k_cache, v_cache, idx_raw, blocks;  // state
        fc::DeviceBuffer conv_snap, S_snap;  // DeltaNet states after each token of a step but the last (rollback)
        int kv = -1;                         // the layer's index in the streamed KV cache (-1: k_cache/v_cache in VRAM)
    };
    std::vector<Layer> layers;
    fc::DeviceWeight output, out_hc_down, out_hc_up;
    fc::DeviceBuffer out_hc_norm, ple_hist, rope_freq;
    const GgufTensor * tok_embd = nullptr;
    const GgufTensor * ple_table = nullptr;

    // activations, sized for kMaxTokens
    fc::DeviceBuffer res, x, xn, lo, gate, mixed, inject, qkv, z, beta, alpha, conv, dn_out, out, qfull, q, qgate, k, v, att, kraw,
        rlogits, ids, wts, sh_g, sh_u, sh_h, sd, sg, moe, ple_emb, ple_key_out, ple_val_out, ple_gated, ple_norm, ple_gates, logits,
        attn_work;
    int cap = fc::kMaxTokens;  // tokens per pass: max(kMaxTokens, prefill_chunk)
    std::unique_ptr<Pinned<float>> h_x, h_ple, h_mixed, h_moe, h_w;
    std::unique_ptr<Pinned<std::int32_t>> h_ids;
    std::unique_ptr<Pinned<std::uint8_t>> h_oncpu;
    fc::DeviceBuffer qi, cells, n_cells, sel_work;  // QSA indexer queries and selections
    // raw indexer keys are needed only until their block of 4 is complete: a ring of a step's worth
    std::int64_t raw_ring() const { return std::max<std::int64_t>(fc::kMaxTokens, opt.prefill_chunk) + 2 * fc::kQsaRatio; }
    std::unique_ptr<fc::Gemm> gemm;
    // KV streaming of the attention layers (null: their K/V are in VRAM), and the staging pool deep prompt
    // chunks use when the prompt cannot borrow one
    std::unique_ptr<KvStreamCache> kv;
    fc::DeviceBuffer kv_stage_own;
    bool kv_streaming() const { return opt.kv_stream > 0 || (opt.kv_stream < 0 && opt.max_ctx > opt.kv_resident); }

    // the token at every position whose keys and values are in the caches; the first n_past are the
    // current sequence, later ones are left over from a sequence that was abandoned by restore()
    std::vector<std::int32_t> history;
    std::int64_t n_past = 0;

    // VRAM expert cache: per layer a pool of slots and the expert -> slot map (host and device)
    struct LayerCache {
        fc::ExpertLayout lay;
        fc::DeviceBuffer pool, dmap;
        std::vector<std::int32_t> map;
    };
    std::vector<LayerCache> cache;
    // MTP head: one more layer (attention + MoE, every expert in VRAM) fed with the main model's last
    // hidden streams and the next token's embedding. Its KV cache holds positions [0, pos) computed
    // from the main model's hidden states; draft positions beyond are scratch, overwritten later.
    // The layer attends to its last 2051 positions only, so its K/V is a ring of `ring` rows (position
    // p at row p % ring) rather than max_ctx rows.
    struct Mtp {
        std::unique_ptr<GgufModel> gguf;
        Layer L;
        fc::DeviceWeight eh_proj, head_down, head_up;
        fc::DeviceBuffer enorm, hnorm, head_norm;
        LayerCache experts;  // all 512, slot = expert id
        fc::DeviceBuffer step, pending_h, h_in, e_in, en, hn, cat, res, dtok;
        std::unique_ptr<Pinned<float>> h_e;
        Pinned<std::int64_t> h_step{1};
        Pinned<std::int32_t> h_tok{1};
        std::int64_t pos = 0;
        std::int64_t ring = 0;
        cudaGraphExec_t graph = nullptr;  // the single-token draft pass
        ~Mtp() {
            if (graph) cudaGraphExecDestroy(graph);
        }
    };
    std::unique_ptr<Mtp> mtp;
    // the main model's hidden rows of the last step (in res): rows [0, rows_valid) at positions rows_pos0..
    std::int64_t rows_pos0 = 0;
    int rows_valid = 0;
    // rollback of the last step (when MTP is on and the step had at most kMaxTokens tokens)
    bool snaps_valid = false;
    int last_T = 0;
    fc::DeviceBuffer ple_hist_prev;

    // pinned images of every expert in the GPU layout, [layer][expert]; device addresses for zero-copy
    std::vector<std::uint8_t *> images, images_d;
    fc::DeviceBuffer host_slots, ypairs_host, plan_oncpu;
    cudaStream_t stream2 = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    std::vector<std::int64_t> counts;  // routing counts [layer][expert]
    // Routing of recent tokens, decayed with a half-life of kRecentHalfLife tokens: a prompt predicts
    // the experts its continuation will use far better than counts gathered on other text.
    static constexpr double kRecentHalfLife = 2048;
    std::vector<double> recent;        // [layer][expert], in units of recent_unit
    double recent_unit = 1.0;          // weight of an observation made now
    std::int64_t tokens_since_adapt = 0;
    std::int64_t pairs_at_adapt = 0, hits_at_adapt = 0;  // expert counters at the last re-ranking
    std::unique_ptr<Pinned<std::uint8_t>> swap_staging;
    static constexpr int kSwapBatch = 64;
    fc::DeviceBuffer slots, eh, ypairs;   // decode steps: one row per (token, expert) pair
    fc::DeviceBuffer gpu_sum, batch_ws;   // prompt chunks: the cached experts' sum per token, and the batch kernels' workspace

    // step state read by the kernels: {pos0, seq}, uploaded at the start of every step
    fc::DeviceBuffer d_step, d_error;
    Pinned<std::int64_t> h_step{2};
    std::int64_t seq = 0;
    std::unique_ptr<Pinned<float>> h_logits;  // [kMaxTokens][n_vocab]
    // CUDA-graph mode: one graph per token count, experts handed to the CPU through mapped memory
    fc::ExpertLink * link_h = nullptr;
    fc::ExpertLink * link_d = nullptr;
    cudaGraphExec_t graphs[fc::kMaxTokens + 1] = {};
    bool graph_mode = false;

    Impl(const GgufModel & m, EngineOptions o) : model(m), cfg(ReferenceConfig::from_gguf(m)), opt(o) {
        validate();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        check(cudaEventCreateWithFlags(&ev_router, cudaEventDisableTiming), "event");
        check(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking), "stream");
        check(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming), "event");
        check(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming), "event");
        fc::init_kernels();
        fc::qsa_init();
        fc::experts_batch_init();
        load();
        allocate();
        if (!opt.mtp_path.empty()) load_mtp();
        CpuExpertsConfig ec;
        ec.threads = opt.cpu_threads;
        ec.precise_activations = opt.precise_cpu_experts;
        ec.pin_threads = opt.pin_cpu_threads;
        experts = std::make_unique<CpuExperts>(model, ec);
        counts.assign(std::size_t(cfg.n_layer) * fc::kExperts, 0);
        recent.assign(counts.size(), 0.0);
        if (!opt.routing_stats.empty()) load_routing(opt.routing_stats);
        fill_cache();
        release_experts();  // CpuExperts and the VRAM cache have their own copies now
        warm_up_cpu_experts();
        // every token reads 16 random rows of the 27 GB PLE table: warm the OS file cache in the
        // background so that early tokens do not wait on disk reads
        if (ple_table) prefetch_mapped(ple_table->data, ple_table->bytes);
        if (opt.host_expert_images) {
            build_images();
            release_experts();
        }
        reset();
    }

    // ---------------------------------------------------------------------------------------------
    // routing statistics: "FNRS", u32 layers, u32 experts, then i64 counts

    void load_routing(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return;  // first run: nothing recorded yet
        char magic[4];
        std::uint32_t nl = 0, ne = 0;
        f.read(magic, 4);
        f.read(reinterpret_cast<char *>(&nl), 4);
        f.read(reinterpret_cast<char *>(&ne), 4);
        if (!f || std::memcmp(magic, "FNRS", 4) != 0 || nl != std::uint32_t(cfg.n_layer) || ne != std::uint32_t(fc::kExperts))
            throw std::runtime_error("engine: " + path + " is not a routing statistics file for this model");
        f.read(reinterpret_cast<char *>(counts.data()), std::streamsize(counts.size() * sizeof(std::int64_t)));
        if (!f) throw std::runtime_error("engine: " + path + " is truncated");
        for (std::int64_t & c : counts) c = std::max<std::int64_t>(c, 0);
    }

    void save_routing(const std::string & path) const {
        std::ofstream f(path, std::ios::binary);
        const std::uint32_t nl = std::uint32_t(cfg.n_layer), ne = fc::kExperts;
        f.write("FNRS", 4);
        f.write(reinterpret_cast<const char *>(&nl), 4);
        f.write(reinterpret_cast<const char *>(&ne), 4);
        f.write(reinterpret_cast<const char *>(counts.data()), std::streamsize(counts.size() * sizeof(std::int64_t)));
        if (!f) throw std::runtime_error("engine: could not write " + path);
    }

    void observe(int il, int e) {
        const std::size_t i = std::size_t(il) * fc::kExperts + std::size_t(e);
        ++counts[i];
        recent[i] += recent_unit;
    }

    void decay(int T) {
        recent_unit *= std::exp2(double(T) / kRecentHalfLife);
        if (recent_unit > 1e100) {
            for (double & r : recent) r /= recent_unit;
            recent_unit = 1.0;
        }
    }

    // Replaces cached experts that recent routing values less than an uncached one (with some
    // hysteresis), at most max_swaps of them. Runs between steps, so no kernel reads the pools.
    void adapt_cache(int max_swaps) {
        struct Swap {
            int il, in, out, slot;
        };
        std::vector<Swap> swaps;
        for (int il = 0; il < cfg.n_layer && int(swaps.size()) < max_swaps; ++il) {
            LayerCache & C = cache[std::size_t(il)];
            const std::size_t base = std::size_t(il) * fc::kExperts;
            double rmass = 0, cmass = 0;
            for (int e = 0; e < fc::kExperts; ++e) {
                rmass += recent[base + std::size_t(e)];
                cmass += double(counts[base + std::size_t(e)]);
            }
            if (rmass <= 0) continue;
            // recent routing, plus the long-run counts at half weight
            auto score = [&](int e) {
                return recent[base + std::size_t(e)] / rmass + 0.5 * double(counts[base + std::size_t(e)]) / std::max(cmass, 1.0);
            };
            std::vector<int> in, out;
            for (int e = 0; e < fc::kExperts; ++e) (C.map[std::size_t(e)] >= 0 ? out : in).push_back(e);
            if (out.empty()) continue;
            std::sort(in.begin(), in.end(), [&](int a, int b) { return score(a) > score(b); });
            std::sort(out.begin(), out.end(), [&](int a, int b) { return score(a) < score(b); });
            for (std::size_t i = 0; i < in.size() && i < out.size() && int(swaps.size()) < max_swaps; ++i) {
                if (score(in[i]) <= 1.25 * score(out[i]) + 1e-6) break;
                swaps.push_back({il, in[i], out[i], C.map[std::size_t(out[i])]});
            }
        }
        if (swaps.empty()) return;
        const auto t_adapt = clk::now();
        check(cudaStreamSynchronize(stream), "adapt");
        std::size_t slot_max = 0;
        for (const LayerCache & C : cache) slot_max = std::max(slot_max, C.lay.slot_bytes);
        if (!swap_staging && images.empty()) swap_staging = std::make_unique<Pinned<std::uint8_t>>(slot_max * kSwapBatch);
        for (std::size_t b0 = 0; b0 < swaps.size(); b0 += kSwapBatch) {
            const std::size_t n = std::min<std::size_t>(kSwapBatch, swaps.size() - b0);
            if (images.empty()) {
                std::vector<std::thread> workers;
                const std::size_t nt = std::min<std::size_t>(16, n);
                for (std::size_t t = 0; t < nt; ++t)
                    workers.emplace_back([&, t] {
                        for (std::size_t j = t; j < n; j += nt) {
                            const Swap & sw = swaps[b0 + j];
                            pack_slot(sw.il, sw.in, swap_staging->get() + j * slot_max);
                        }
                    });
                for (auto & w : workers) w.join();
            }
            for (std::size_t j = 0; j < n; ++j) {
                const Swap & sw = swaps[b0 + j];
                LayerCache & C = cache[std::size_t(sw.il)];
                const std::uint8_t * src = images.empty() ? swap_staging->get() + j * slot_max
                                                          : images[std::size_t(sw.il)] + std::size_t(sw.in) * C.lay.slot_bytes;
                check(cudaMemcpyAsync(C.pool.as<std::uint8_t>() + std::size_t(sw.slot) * C.lay.slot_bytes, src, C.lay.slot_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "expert swap");
            }
            check(cudaStreamSynchronize(stream), "expert swap");
        }
        std::vector<bool> touched(std::size_t(cfg.n_layer), false);
        for (const Swap & sw : swaps) {
            LayerCache & C = cache[std::size_t(sw.il)];
            C.map[std::size_t(sw.out)] = -1;
            C.map[std::size_t(sw.in)] = sw.slot;
            touched[std::size_t(sw.il)] = true;
        }
        for (int il = 0; il < cfg.n_layer; ++il)
            if (touched[std::size_t(il)]) {
                LayerCache & C = cache[std::size_t(il)];
                check(cudaMemcpy(C.dmap.get(), C.map.data(), C.dmap.bytes(), cudaMemcpyHostToDevice), "expert map");
            }
        stats.cache_swaps += std::int64_t(swaps.size());
        stats.cache_swap_ms += std::chrono::duration<double, std::milli>(clk::now() - t_adapt).count();
    }

    void load_mtp() {
        mtp = std::make_unique<Mtp>();
        Mtp & M = *mtp;
        M.gguf = std::make_unique<GgufModel>(std::vector<std::string>{opt.mtp_path});
        const GgufModel & g = *M.gguf;
        const int il = cfg.n_layer, E = fc::kEmbd, HCD = fc::kHcd, R = fc::kHcRank;
        if (!g.has("qwen4exp.nextn_predict_layers") || g.get_int("qwen4exp.nextn_predict_layers") != 1)
            throw std::runtime_error("engine: " + opt.mtp_path + " is not a single-layer qwen4exp MTP head");
        const std::string b = "blk." + std::to_string(il) + ".";
        auto W = [&](const std::string & name, int k, int n) {
            fc::DeviceWeight w = fc::upload_gemv_weight(g.tensor(name));
            if (w.view.k != k || w.view.n != n) throw std::runtime_error("engine: unexpected shape for MTP " + name);
            return w;
        };
        auto V = [&](const std::string & name, std::int64_t n) { return upload_f32(g.tensor(name), n); };
        Layer & L = M.L;
        L.hc_attn_norm = V(b + "hc_attn_norm.weight", HCD);
        L.hc_ffn_norm = V(b + "hc_ffn_norm.weight", HCD);
        L.hc_attn_down = W(b + "hc_attn_down.weight", HCD, R);
        L.hc_attn_up = W(b + "hc_attn_up.weight", R, HCD);
        L.hc_attn_inject = W(b + "hc_attn_inject.weight", HCD, fc::kHc);
        L.hc_ffn_down = W(b + "hc_ffn_down.weight", HCD, R);
        L.hc_ffn_up = W(b + "hc_ffn_up.weight", R, HCD);
        L.hc_ffn_inject = W(b + "hc_ffn_inject.weight", HCD, fc::kHc);
        L.wq = W(b + "attn_q.weight", E, 2 * fc::kHeads * fc::kHeadDim);
        L.wk = W(b + "attn_k.weight", E, fc::kKvHeads * fc::kHeadDim);
        L.wv = W(b + "attn_v.weight", E, fc::kKvHeads * fc::kHeadDim);
        L.wo = W(b + "attn_output.weight", fc::kHeads * fc::kHeadDim, E);
        L.q_norm = V(b + "attn_q_norm.weight", fc::kHeadDim);
        L.k_norm = V(b + "attn_k_norm.weight", fc::kHeadDim);
        L.router = W(b + "ffn_gate_inp.weight", E, fc::kExperts);
        L.sh_gate = W(b + "ffn_gate_shexp.weight", E, fc::kFfShared);
        L.sh_up = W(b + "ffn_up_shexp.weight", E, fc::kFfShared);
        L.sh_down = W(b + "ffn_down_shexp.weight", fc::kFfShared, E);
        L.sh_gate_inp = W(b + "ffn_gate_inp_shexp.weight", E, 1);
        M.eh_proj = W(b + "nextn.eh_proj.weight", 2 * E, E);
        M.enorm = V(b + "nextn.enorm.weight", E);
        M.hnorm = V(b + "nextn.hnorm.weight", HCD);
        M.head_norm = V(b + "nextn.hc_head_norm.weight", HCD);
        M.head_down = W(b + "nextn.hc_head_down.weight", HCD, R);
        M.head_up = W(b + "nextn.hc_head_up.weight", R, HCD);
        // A pass of T <= cap tokens writes rows pos0 .. pos0+T-1 and reads the 2050 before each: with cap + 2051
        // rows no two of those positions share a row.
        M.ring = (std::int64_t(cap) + fc::kQsaWidth + 255) / 256 * 256;
        const std::size_t kv = std::size_t(M.ring) * fc::kKvHeads * fc::kHeadDim * sizeof(half);
        L.k_cache = zeros(kv);
        L.v_cache = zeros(kv);
        // every expert in VRAM: the MTP layer has no CPU path
        const GgufTensor & ge = g.tensor(b + "ffn_gate_exps.weight");
        const GgufTensor & ue = g.tensor(b + "ffn_up_exps.weight");
        const GgufTensor & de = g.tensor(b + "ffn_down_exps.weight");
        const std::vector<std::int64_t> gate_shape{E, fc::kExpertFF, fc::kExperts}, down_shape{fc::kExpertFF, E, fc::kExperts};
        if (ge.shape != gate_shape || ue.shape != gate_shape || de.shape != down_shape || ue.type != ge.type)
            throw std::runtime_error("engine: unexpected MTP expert tensors in " + opt.mtp_path);
        M.experts.lay = fc::expert_layout(ge.type, de.type);
        M.experts.map.resize(fc::kExperts);
        M.experts.pool = fc::DeviceBuffer(std::size_t(fc::kExperts) * M.experts.lay.slot_bytes);
        {
            Pinned<std::uint8_t> staging(std::size_t(64) * M.experts.lay.slot_bytes);
            for (int e0 = 0; e0 < fc::kExperts; e0 += 64) {
                std::vector<std::thread> workers;
                for (int t = 0; t < 16; ++t)
                    workers.emplace_back([&, t] {
                        for (int e = e0 + t; e < e0 + 64; e += 16)
                            fc::pack_expert(M.experts.lay, ge, ue, de, e, staging.get() + std::size_t(e - e0) * M.experts.lay.slot_bytes);
                    });
                for (auto & w : workers) w.join();
                check(cudaMemcpy(M.experts.pool.as<std::uint8_t>() + std::size_t(e0) * M.experts.lay.slot_bytes, staging.get(),
                                 std::size_t(64) * M.experts.lay.slot_bytes, cudaMemcpyHostToDevice),
                      "MTP experts");
            }
        }
        for (int e = 0; e < fc::kExperts; ++e) M.experts.map[std::size_t(e)] = e;
        M.experts.dmap = fc::DeviceBuffer(fc::kExperts * sizeof(std::int32_t));
        check(cudaMemcpy(M.experts.dmap.get(), M.experts.map.data(), M.experts.dmap.bytes(), cudaMemcpyHostToDevice), "MTP map");
        const std::size_t T = std::size_t(cap), f = sizeof(float);
        M.step = zeros(sizeof(std::int64_t));
        M.pending_h = zeros(HCD * f);
        M.h_in = fc::DeviceBuffer(T * HCD * f);
        M.e_in = fc::DeviceBuffer(T * E * f);
        M.en = fc::DeviceBuffer(T * E * f);
        M.hn = fc::DeviceBuffer(T * HCD * f);
        M.cat = fc::DeviceBuffer(T * fc::kHc * 2 * E * f);
        M.res = fc::DeviceBuffer(T * HCD * f);
        M.dtok = fc::DeviceBuffer(sizeof(std::int32_t));
        M.h_e = std::make_unique<Pinned<float>>(T * E);
        ple_hist_prev = fc::DeviceBuffer(ple_hist.bytes());
    }

    // One pass of the MTP layer over T (hidden, token) pairs at positions pos0.. (h_in and the
    // embeddings in M.h_e are filled by the caller). With want_draft, the head's argmax for the last
    // pair is returned.
    std::int32_t mtp_pass(int T, std::int64_t pos0, bool want_draft) {
        Mtp & M = *mtp;
        M.h_step.get()[0] = pos0;
        if (T == 1 && want_draft && opt.cuda_graphs) {
            // a draft pass is replayed as one CUDA graph (positions and inputs come from pinned memory)
            if (!M.graph) {
                cudaGraph_t g = nullptr;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "MTP capture");
                try {
                    mtp_enqueue(1, true);
                } catch (...) {
                    cudaStreamEndCapture(stream, &g);
                    if (g) cudaGraphDestroy(g);
                    throw;
                }
                check(cudaStreamEndCapture(stream, &g), "MTP capture");
                const cudaError_t err = cudaGraphInstantiate(&M.graph, g, 0);
                cudaGraphDestroy(g);
                check(err, "MTP graph");
            }
            check(cudaGraphLaunch(M.graph, stream), "MTP graph");
        } else {
            mtp_enqueue(T, want_draft);
        }
        if (!want_draft) return -1;
        check(cudaStreamSynchronize(stream), "draft");
        return M.h_tok.get()[0];
    }

    void mtp_enqueue(int T, bool want_draft) {
        Mtp & M = *mtp;
        Layer & L = M.L;
        const std::int64_t * pos = M.step.as<std::int64_t>();
        check(cudaMemcpyAsync(M.step.get(), M.h_step.get(), sizeof(std::int64_t), cudaMemcpyHostToDevice, stream), "MTP step");
        check(cudaMemcpyAsync(M.e_in.get(), M.h_e->get(), std::size_t(T) * fc::kEmbd * sizeof(float), cudaMemcpyHostToDevice, stream), "MTP embed");
        // inputs: per-stream RMSNorm of the hidden streams, RMSNorm of the embedding, concatenated per stream
        fc::hc_norm(M.h_in.as<float>(), M.hnorm.as<float>(), M.hn.as<float>(), T, cfg.rms_eps, stream);
        fc::rms_norm_rows(M.e_in.as<float>(), M.enorm.as<float>(), M.en.as<float>(), T, fc::kEmbd, cfg.rms_eps, stream);
        fc::mtp_concat(M.en.as<float>(), M.hn.as<float>(), M.cat.as<float>(), T, stream);
        linear(M.eh_proj, M.cat, M.res, T * fc::kHc);  // [T][4] rows of 5120 -> the layer's residual streams

        hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, T, 0, M.res.as<float>());
        linear_multi({{&L.wq, &qfull}, {&L.wk, &k}, {&L.wv, &v}}, mixed, T);
        fc::KvStore ring;
        ring.k = L.k_cache.as<half>();
        ring.v = L.v_cache.as<half>();
        ring.ring = M.ring;
        fc::attn_prep(qfull.as<float>(), k.as<float>(), v.as<float>(), L.q_norm.as<float>(), L.k_norm.as<float>(), rope_freq.as<double>(),
                      q.as<float>(), qgate.as<float>(), ring, pos, T, cfg.rms_eps, stream);
        // the most recent 2051 positions (llama.cpp attends densely; drafts are verified either way), as ring rows
        fc::window_cells(pos, T, fc::kQsaWidth, cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), stream, M.ring);
        fc::attn_sparse(q.as<float>(), qgate.as<float>(), L.k_cache.as<half>(), L.v_cache.as<half>(), cells.as<std::int32_t>(),
                        n_cells.as<std::int32_t>(), T, cfg.kq_scale, attn_work.as<float>(), att.as<float>(), stream);
        linear(L.wo, att, out, T);
        fc::hc_combine(M.res.as<float>(), out.as<float>(), inject.as<float>(), T, stream);

        hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, T, 0, M.res.as<float>());
        linear(L.router, mixed, rlogits, T);
        fc::router_topk(rlogits.as<float>(), ids.as<std::int32_t>(), wts.as<float>(), T, stream);
        fc::moe_slots(ids.as<std::int32_t>(), M.experts.dmap.as<std::int32_t>(), slots.as<std::int32_t>(), T, stream);
        linear_multi({{&L.sh_gate, &sh_g}, {&L.sh_up, &sh_u}, {&L.sh_gate_inp, &sg}}, mixed, T);
        fc::swiglu(sh_g.as<float>(), sh_u.as<float>(), sh_h.as<float>(), T * fc::kFfShared, stream);
        linear(L.sh_down, sh_h, sd, T);
        check(cudaMemsetAsync(moe.get(), 0, std::size_t(T) * fc::kEmbd * sizeof(float), stream), "MTP moe");
        for (int t0 = 0; t0 < T; t0 += fc::kMaxTokens) {  // the per-pair expert kernels, kMaxTokens tokens at a time
            const int n = std::min(fc::kMaxTokens, T - t0);
            const std::size_t xo = std::size_t(t0) * fc::kEmbd, ko = std::size_t(t0) * fc::kUsed;
            fc::experts_gpu(M.experts.lay, M.experts.pool.as<std::uint8_t>(), slots.as<std::int32_t>() + ko, wts.as<float>() + ko,
                            mixed.as<float>() + xo, eh.as<float>(), ypairs.as<float>(), n, stream);
            fc::moe_combine(ypairs.as<float>(), nullptr, moe.as<float>() + xo, sd.as<float>() + xo, sg.as<float>() + t0,
                            out.as<float>() + xo, n, stream);
        }
        fc::hc_combine(M.res.as<float>(), out.as<float>(), inject.as<float>(), T, stream);
        if (!want_draft) return;
        hc_mix(M.head_norm, M.head_down, M.head_up, nullptr, 1, T - 1, M.res.as<float>());
        gemv(output, mixed, logits, 1);
        fc::argmax(logits.as<float>(), cfg.n_vocab, M.dtok.as<std::int32_t>(), stream);
        check(cudaMemcpyAsync(M.h_tok.get(), M.dtok.get(), sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream), "draft");
    }

    void mtp_embed(int t, std::int32_t token) {
        const std::size_t rb = row_bytes(tok_embd->type, fc::kEmbd);
        dequantize_row(tok_embd->type, tok_embd->data + std::size_t(token) * rb, mtp->h_e->get() + std::size_t(t) * fc::kEmbd, fc::kEmbd);
    }

    // Brings the MTP cache up to the main model's tokens: pairs (h of position q-1, token at q) for the
    // positions not yet covered, using the hidden rows of the last step. Runs before every step.
    void mtp_catchup() {
        if (!mtp) return;
        Mtp & M = *mtp;
        if (rows_valid <= 0) return;
        const std::int64_t end = rows_pos0 + rows_valid;
        if (M.pos < rows_pos0) M.pos = rows_pos0;  // rows lost (e.g. after restore): a gap in the MTP cache
        const int n = int(end - M.pos);
        const std::size_t hb = fc::kHcd * sizeof(float);
        if (n > 0) {
            for (int i = 0; i < n; ++i) {
                const std::int64_t q = M.pos + i;
                const float * src = q - 1 < rows_pos0 ? M.pending_h.as<float>() : res.as<float>() + std::size_t(q - 1 - rows_pos0) * fc::kHcd;
                check(cudaMemcpyAsync(M.h_in.as<float>() + std::size_t(i) * fc::kHcd, src, hb, cudaMemcpyDeviceToDevice, stream), "MTP h");
                mtp_embed(i, history[std::size_t(q)]);
            }
            mtp_pass(n, M.pos, false);
            M.pos = end;
        }
        // the next pair starts from the hidden state of the last kept position
        check(cudaMemcpyAsync(M.pending_h.get(), res.as<float>() + std::size_t(rows_valid - 1) * fc::kHcd, hb, cudaMemcpyDeviceToDevice, stream),
              "MTP h");
        rows_pos0 = end;
        rows_valid = 0;
    }

    std::vector<std::int32_t> draft(std::int32_t next, int k) {
        if (!mtp) throw std::runtime_error("engine: no MTP head loaded");
        if (next < 0 || next >= cfg.n_vocab) throw std::runtime_error("engine: token id out of range");
        mtp_catchup();
        Mtp & M = *mtp;
        std::vector<std::int32_t> out;
        const std::size_t hb = fc::kHcd * sizeof(float);
        std::int32_t tok = next;
        for (int i = 0; i < k; ++i) {
            // the first pair uses the main model's hidden state; later ones the MTP layer's own output
            const float * src = i == 0 ? M.pending_h.as<float>() : M.res.as<float>();
            check(cudaMemcpyAsync(M.h_in.get(), src, hb, cudaMemcpyDeviceToDevice, stream), "MTP h");
            mtp_embed(0, tok);
            tok = mtp_pass(1, n_past + i, true);
            out.push_back(tok);
        }
        // the entry at n_past (true hidden state, the token that will be fed) is final
        M.pos = n_past + 1;
        return out;
    }

    void rollback(int n_keep) {
        if (!snaps_valid || n_keep < 1 || n_keep > last_T) throw std::runtime_error("engine: nothing to roll back to");
        if (n_keep < last_T) {
            for (Layer & L : layers) {
                if (!L.recurrent) continue;
                check(cudaMemcpyAsync(L.S.get(), L.S_snap.as<std::uint8_t>() + std::size_t(n_keep - 1) * L.S.bytes(), L.S.bytes(),
                                      cudaMemcpyDeviceToDevice, stream),
                      "rollback");
                check(cudaMemcpyAsync(L.conv_state.get(), L.conv_snap.as<std::uint8_t>() + std::size_t(n_keep - 1) * L.conv_state.bytes(),
                                      L.conv_state.bytes(), cudaMemcpyDeviceToDevice, stream),
                      "rollback");
            }
            if (ple_table) fc::ple_hist_rebuild(ple_hist_prev.as<float>(), ple_norm.as<float>(), ple_hist.as<float>(), n_keep, stream);
            n_past -= last_T - n_keep;
            rows_valid = n_keep;
            if (mtp && mtp->pos > n_past) mtp->pos = n_past;
        }
        snaps_valid = false;
    }

    // Slot image of expert e of layer il, rebuilt from the CPU's resident copy (bit-exact, and it never
    // touches the memory-mapped GGUF, whose pages may have been evicted to disk).
    void pack_slot(int il, int e, std::uint8_t * dst) {
        const LayerCache & C = cache[std::size_t(il)];
        const CpuExperts::ExportSizes sz = experts->export_sizes(il);
        thread_local std::vector<std::uint8_t> tmp;
        tmp.resize(2 * sz.gate_up + sz.down);
        experts->export_expert(il, e, tmp.data(), tmp.data() + sz.gate_up, tmp.data() + 2 * sz.gate_up);
        fc::pack_expert_rows(C.lay, tmp.data(), tmp.data() + sz.gate_up, tmp.data() + 2 * sz.gate_up, dst);
    }

    // One CPU expert call per layer and token count on dummy data, so that the first real step does not
    // pay for page faults and lazily allocated scratch while the GPU is waiting on it (with a deadline).
    void warm_up_cpu_experts() {
        const int T = fc::kMaxTokens;
        std::vector<float> xw(std::size_t(T) * fc::kEmbd, 0.01f), w(std::size_t(T) * fc::kUsed, 0.1f), o(std::size_t(T) * fc::kEmbd);
        std::vector<std::int32_t> idw(std::size_t(T) * fc::kUsed);
        for (std::size_t i = 0; i < idw.size(); ++i) idw[i] = std::int32_t((i * 37) % fc::kExperts);
        for (int il = 0; il < cfg.n_layer; ++il)
            for (int t = 1; t <= T; ++t) experts->run(il, t, xw.data(), idw.data(), w.data(), nullptr, o.data());
    }

    void release_experts() {
        for (int il = 0; il < cfg.n_layer; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const GgufTensor & t = model.tensor(b + n);
                release_mapped(t.data, t.bytes);
            }
        }
    }

    // Packs every expert into pinned host memory in the GPU layout, if the RAM is there.
    void build_images() {
        std::uint64_t need = 0;
        for (const LayerCache & C : cache) need += std::uint64_t(C.lay.slot_bytes) * fc::kExperts;
        const std::uint64_t margin = std::uint64_t(16) << 30;
        if (free_ram_bytes() < need + margin) {
            std::fprintf(stderr, "engine: %.1f GiB of free RAM is not enough for the pinned expert images (%.1f GiB); misses stay on the CPU\n",
                         free_ram_bytes() / 1073741824.0, need / 1073741824.0);
            return;
        }
        images.assign(cache.size(), nullptr);
        images_d.assign(cache.size(), nullptr);
        for (int il = 0; il < cfg.n_layer; ++il) {
            LayerCache & C = cache[std::size_t(il)];
            const std::size_t bytes = C.lay.slot_bytes * fc::kExperts;
            check(cudaHostAlloc(reinterpret_cast<void **>(&images[std::size_t(il)]), bytes, cudaHostAllocMapped | cudaHostAllocPortable),
                  "expert images");
            check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&images_d[std::size_t(il)]), images[std::size_t(il)], 0), "expert images");
            std::vector<std::thread> workers;
            for (int t = 0; t < 16; ++t)
                workers.emplace_back([&, t] {
                    for (int e = t; e < fc::kExperts; e += 16) pack_slot(il, e, images[std::size_t(il)] + std::size_t(e) * C.lay.slot_bytes);
                });
            for (auto & w : workers) w.join();
        }
    }

    // Fills the free VRAM with the experts of highest routing count per byte.
    void fill_cache() {
        const int nl = cfg.n_layer;
        cache.resize(std::size_t(nl));
        for (int il = 0; il < nl; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            LayerCache & C = cache[std::size_t(il)];
            C.lay = fc::expert_layout(model.tensor(b + "ffn_gate_exps.weight").type, model.tensor(b + "ffn_down_exps.weight").type);
            C.map.assign(fc::kExperts, -1);
            C.dmap = fc::DeviceBuffer(fc::kExperts * sizeof(std::int32_t));
        }
        std::size_t free_b = 0, total_b = 0;
        check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const std::int64_t budget = opt.expert_cache_mib >= 0 ? opt.expert_cache_mib << 20
                                                              : std::int64_t(free_b) - (opt.vram_reserve_mib << 20) - (std::int64_t(256) << 20);
        struct Cand {
            double score;
            int il, e;
        };
        std::vector<Cand> cand;
        for (int il = 0; il < nl; ++il)
            for (int e = 0; e < fc::kExperts; ++e)
                cand.push_back({double(counts[std::size_t(il) * fc::kExperts + e] + 1) / double(cache[std::size_t(il)].lay.slot_bytes), il, e});
        std::stable_sort(cand.begin(), cand.end(), [](const Cand & a, const Cand & b) { return a.score > b.score; });
        std::vector<std::vector<int>> chosen(static_cast<std::size_t>(nl));
        std::int64_t used = 0;
        for (const Cand & c : cand) {
            const std::int64_t sb = std::int64_t(cache[std::size_t(c.il)].lay.slot_bytes);
            if (used + sb > budget) continue;
            chosen[std::size_t(c.il)].push_back(c.e);
            used += sb;
        }
        std::size_t most = 0;
        for (int il = 0; il < nl; ++il) most = std::max(most, chosen[std::size_t(il)].size() * cache[std::size_t(il)].lay.slot_bytes);
        std::unique_ptr<Pinned<std::uint8_t>> staging = most ? std::make_unique<Pinned<std::uint8_t>>(most) : nullptr;
        for (int il = 0; il < nl; ++il) {
            LayerCache & C = cache[std::size_t(il)];
            std::vector<int> & list = chosen[std::size_t(il)];
            std::sort(list.begin(), list.end());
            if (!list.empty()) {
                const int nt = std::max(1, std::min(16, int(list.size())));
                std::vector<std::thread> pool;
                for (int t = 0; t < nt; ++t)
                    pool.emplace_back([&, t] {
                        for (std::size_t j = std::size_t(t); j < list.size(); j += std::size_t(nt))
                            pack_slot(il, list[j], staging->get() + j * C.lay.slot_bytes);
                    });
                for (auto & th : pool) th.join();
                C.pool = fc::DeviceBuffer(list.size() * C.lay.slot_bytes);
                check(cudaMemcpy(C.pool.get(), staging->get(), C.pool.bytes(), cudaMemcpyHostToDevice), "expert cache");
                for (std::size_t j = 0; j < list.size(); ++j) C.map[std::size_t(list[j])] = std::int32_t(j);
            }
            check(cudaMemcpy(C.dmap.get(), C.map.data(), C.dmap.bytes(), cudaMemcpyHostToDevice), "expert map");
            stats.cached_experts += std::int64_t(list.size());
        }
        stats.cache_gib = double(used) / double(1 << 30);
    }
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        for (cudaGraphExec_t g : graphs)
            if (g) cudaGraphExecDestroy(g);
        if (link_h) cudaFreeHost(link_h);
        for (std::uint8_t * p : images)
            if (p) cudaFreeHost(p);
        if (ev_fork) cudaEventDestroy(ev_fork);
        if (ev_join) cudaEventDestroy(ev_join);
        if (stream2) cudaStreamDestroy(stream2);
        if (ev_router) cudaEventDestroy(ev_router);
        if (stream) cudaStreamDestroy(stream);
    }

    void validate() const {
        require(cfg.n_embd == fc::kEmbd && cfg.hc == fc::kHc && cfg.hc_rank == fc::kHcRank, "hidden size / hyper-connections");
        require(cfg.n_head == fc::kHeads && cfg.n_head_kv == fc::kKvHeads && cfg.head_dim == fc::kHeadDim && cfg.n_rot == fc::kRot,
                "attention heads");
        require(cfg.ssm_state == fc::kDnState && cfg.ssm_k_heads == fc::kDnKHeads && cfg.ssm_v_heads == fc::kDnVHeads &&
                    cfg.ssm_conv == fc::kDnConv,
                "Gated DeltaNet");
        require(cfg.n_expert == fc::kExperts && cfg.n_expert_used == fc::kUsed && cfg.n_ff_shexp == fc::kFfShared, "MoE");
        require(cfg.idx_head_dim == fc::kIdxDim && cfg.idx_n_head == fc::kQsaHeads && cfg.idx_top_k == fc::kQsaTopK, "indexer");
        require(cfg.ple_layer < 0 || (cfg.ple_n_heads == fc::kPleHeads && cfg.ple_head_dim == fc::kPleHeadDim &&
                                      cfg.ple_conv_kernel == fc::kPleKernel && cfg.ple_ngram == fc::kPleDilation),
                "PLE");
        if (opt.max_ctx < 1 || opt.max_ctx > (std::int64_t(1) << 20)) throw std::runtime_error("engine: max_ctx out of range");
        if (opt.prefill_chunk < 1 || opt.prefill_chunk > 8192) throw std::runtime_error("engine: prefill_chunk out of range");
    }

    fc::DeviceWeight W(const std::string & name, int k, int n) const {
        fc::DeviceWeight w = fc::upload_gemv_weight(model.tensor(name));
        if (w.view.k != k || w.view.n != n) throw std::runtime_error("engine: unexpected shape for " + name);
        return w;
    }
    fc::DeviceBuffer V(const std::string & name, std::int64_t n) const { return upload_f32(model.tensor(name), n); }

    void load() {
        const int E = fc::kEmbd, HCD = fc::kHcd, R = fc::kHcRank;
        tok_embd = &model.tensor("token_embd.weight");
        require(tok_embd->shape[0] == E && tok_embd->shape[1] == cfg.n_vocab, "token embedding");
        output = W(model.find("output.weight") ? "output.weight" : "token_embd.weight", E, cfg.n_vocab);
        out_hc_norm = V("output_hc_norm.weight", HCD);
        out_hc_down = W("output_hc_down.weight", HCD, R);
        out_hc_up = W("output_hc_up.weight", R, HCD);
        if (cfg.ple_layer >= 0) {
            ple_table = &model.tensor("per_layer_token_embd.weight");
            require(ple_table->shape[0] == fc::kPleHeadDim, "PLE table");
        }
        layers.resize(std::size_t(cfg.n_layer));
        for (int il = 0; il < cfg.n_layer; ++il) {
            Layer & L = layers[std::size_t(il)];
            const std::string b = "blk." + std::to_string(il) + ".";
            L.recurrent = cfg.recurrent[std::size_t(il)];
            L.hc_attn_norm = V(b + "hc_attn_norm.weight", HCD);
            L.hc_ffn_norm = V(b + "hc_ffn_norm.weight", HCD);
            L.hc_attn_down = W(b + "hc_attn_down.weight", HCD, R);
            L.hc_attn_up = W(b + "hc_attn_up.weight", R, HCD);
            L.hc_attn_inject = W(b + "hc_attn_inject.weight", HCD, fc::kHc);
            L.hc_ffn_down = W(b + "hc_ffn_down.weight", HCD, R);
            L.hc_ffn_up = W(b + "hc_ffn_up.weight", R, HCD);
            L.hc_ffn_inject = W(b + "hc_ffn_inject.weight", HCD, fc::kHc);
            if (L.recurrent) {
                L.wqkv = W(b + "attn_qkv.weight", E, fc::kDnConvDim);
                L.wgate = W(b + "attn_gate.weight", E, fc::kDnVDim);
                L.ssm_beta = W(b + "ssm_beta.weight", E, fc::kDnVHeads);
                L.ssm_alpha = W(b + "ssm_alpha.weight", E, fc::kDnVHeads);
                L.ssm_out = W(b + "ssm_out.weight", fc::kDnVDim, E);
                L.conv1d = V(b + "ssm_conv1d.weight", std::int64_t(fc::kDnConv) * fc::kDnConvDim);
                L.dt = V(b + "ssm_dt.bias", fc::kDnVHeads);
                L.a = V(b + "ssm_a", fc::kDnVHeads);
                L.ssm_norm = V(b + "ssm_norm.weight", fc::kDnState);
                L.conv_state = fc::DeviceBuffer(std::size_t(fc::kDnConv - 1) * fc::kDnConvDim * sizeof(float));
                L.S = fc::DeviceBuffer(std::size_t(fc::kDnVHeads) * fc::kDnState * fc::kDnState * sizeof(float));
                if (!opt.mtp_path.empty()) {
                    L.conv_snap = fc::DeviceBuffer((fc::kMaxTokens - 1) * L.conv_state.bytes());
                    L.S_snap = fc::DeviceBuffer((fc::kMaxTokens - 1) * L.S.bytes());
                }
            } else {
                L.wq = W(b + "attn_q.weight", E, 2 * fc::kHeads * fc::kHeadDim);
                L.wk = W(b + "attn_k.weight", E, fc::kKvHeads * fc::kHeadDim);
                L.wv = W(b + "attn_v.weight", E, fc::kKvHeads * fc::kHeadDim);
                L.wo = W(b + "attn_output.weight", fc::kHeads * fc::kHeadDim, E);
                L.q_norm = V(b + "attn_q_norm.weight", fc::kHeadDim);
                L.k_norm = V(b + "attn_k_norm.weight", fc::kHeadDim);
                L.idx_k = W(b + "indexer.k_proj.weight", E, fc::kIdxDim);
                L.idx_q = W(b + "indexer.q_proj.weight", E, fc::kQsaHeads * fc::kQsaDim);
                L.idx_q_norm = V(b + "indexer.q_norm.weight", fc::kQsaDim);
                L.idx_k_norm = V(b + "indexer.k_norm.weight", fc::kQsaDim);
                require(cfg.compress_ratio[std::size_t(il)] == fc::kQsaRatio, "QSA compress ratio");
                if (kv_streaming()) {
                    if (!kv) kv = std::make_unique<KvStreamCache>(opt.max_ctx, opt.kv_resident, opt.kv_group_tokens, stream);
                    L.kv = kv->add_layer();
                } else {
                    const std::size_t kvb = std::size_t(opt.max_ctx) * fc::kKvHeads * fc::kHeadDim * sizeof(half);
                    L.k_cache = fc::DeviceBuffer(kvb);
                    L.v_cache = fc::DeviceBuffer(kvb);
                }
                L.idx_raw = fc::DeviceBuffer(std::size_t(raw_ring()) * fc::kIdxDim * sizeof(float));
                L.blocks = fc::DeviceBuffer(std::size_t(opt.max_ctx / fc::kQsaRatio + 1) * fc::kQsaDim * sizeof(float));
            }
            if (il == cfg.ple_layer) {
                L.ple = true;
                L.ple_key = W(b + "ple_key.weight", E, HCD);
                L.ple_value = W(b + "ple_value.weight", E, E);
                L.ple_nk = V(b + "ple_norm_key.weight", HCD);
                L.ple_nq = V(b + "ple_norm_query.weight", HCD);
                L.ple_nc = V(b + "ple_norm_conv.weight", HCD);
                L.ple_conv = V(b + "ple_conv1d.weight", std::int64_t(fc::kPleKernel) * HCD);
            }
            L.router = W(b + "ffn_gate_inp.weight", E, fc::kExperts);
            L.sh_gate = W(b + "ffn_gate_shexp.weight", E, fc::kFfShared);
            L.sh_up = W(b + "ffn_up_shexp.weight", E, fc::kFfShared);
            L.sh_down = W(b + "ffn_down_shexp.weight", fc::kFfShared, E);
            L.sh_gate_inp = W(b + "ffn_gate_inp_shexp.weight", E, 1);
        }
        std::vector<double> inv(fc::kRot / 2);
        for (int i = 0; i < fc::kRot / 2; ++i) inv[std::size_t(i)] = std::pow(double(cfg.rope_base), -2.0 * i / fc::kRot);
        rope_freq = fc::DeviceBuffer(inv.size() * sizeof(double));
        check(cudaMemcpy(rope_freq.get(), inv.data(), rope_freq.bytes(), cudaMemcpyHostToDevice), "rope");
    }

    void allocate() {
        cap = std::max(fc::kMaxTokens, opt.prefill_chunk);
        const std::size_t T = std::size_t(cap), f = sizeof(float);
        h_x = std::make_unique<Pinned<float>>(T * fc::kEmbd);
        h_ple = std::make_unique<Pinned<float>>(T * fc::kEmbd);
        h_mixed = std::make_unique<Pinned<float>>(T * fc::kEmbd);
        h_moe = std::make_unique<Pinned<float>>(T * fc::kEmbd);
        h_w = std::make_unique<Pinned<float>>(T * fc::kUsed);
        h_ids = std::make_unique<Pinned<std::int32_t>>(T * fc::kUsed);
        h_oncpu = std::make_unique<Pinned<std::uint8_t>>(T * fc::kUsed);
        qi = fc::DeviceBuffer(T * fc::kQsaHeads * fc::kQsaDim * f);
        cells = fc::DeviceBuffer(T * fc::kQsaWidth * sizeof(std::int32_t));
        n_cells = fc::DeviceBuffer(T * sizeof(std::int32_t));
        sel_work = fc::DeviceBuffer(fc::qsa_select_work_bytes(int(T), opt.max_ctx));
        // the largest dense matrix multiplied in batches: the attention query projection
        gemm = std::make_unique<fc::Gemm>(std::size_t(2) * fc::kHeads * fc::kHeadDim * fc::kEmbd, stream);
        auto buf = [&](std::size_t floats) { return fc::DeviceBuffer(T * floats * f); };
        res = buf(fc::kHcd);
        x = buf(fc::kEmbd);
        xn = buf(fc::kHcd);
        lo = buf(fc::kHcRank);
        gate = buf(fc::kHcd);
        mixed = buf(fc::kEmbd);
        inject = buf(fc::kHc);
        qkv = buf(fc::kDnConvDim);
        z = buf(fc::kDnVDim);
        beta = buf(fc::kDnVHeads);
        alpha = buf(fc::kDnVHeads);
        conv = buf(fc::kDnConvDim);
        dn_out = buf(fc::kDnVDim);
        out = buf(fc::kEmbd);
        qfull = buf(2 * fc::kHeads * fc::kHeadDim);
        q = buf(fc::kHeads * fc::kHeadDim);
        qgate = buf(fc::kHeads * fc::kHeadDim);
        k = buf(fc::kKvHeads * fc::kHeadDim);
        v = buf(fc::kKvHeads * fc::kHeadDim);
        att = buf(fc::kHeads * fc::kHeadDim);
        kraw = buf(fc::kIdxDim);
        rlogits = buf(fc::kExperts);
        ids = buf(fc::kUsed);
        wts = buf(fc::kUsed);
        sh_g = buf(fc::kFfShared);
        sh_u = buf(fc::kFfShared);
        sh_h = buf(fc::kFfShared);
        sd = buf(fc::kEmbd);
        sg = buf(1);
        moe = buf(fc::kEmbd);
        ple_emb = buf(fc::kEmbd);
        ple_key_out = buf(fc::kHcd);
        ple_val_out = buf(fc::kEmbd);
        ple_gated = buf(fc::kHcd);
        ple_norm = buf(fc::kHcd);
        ple_gates = buf(fc::kHc);
        logits = fc::DeviceBuffer(std::size_t(fc::kMaxTokens) * cfg.n_vocab * f);
        attn_work = fc::DeviceBuffer(fc::attn_sparse_work_floats(int(T)) * f);
        slots = fc::DeviceBuffer(T * fc::kUsed * sizeof(std::int32_t));
        d_step = zeros(2 * sizeof(std::int64_t));
        d_error = zeros(sizeof(int));
        h_logits = std::make_unique<Pinned<float>>(std::size_t(fc::kMaxTokens) * cfg.n_vocab);
        const std::size_t link_bytes = std::size_t(cfg.n_layer) * sizeof(fc::ExpertLink);
        check(cudaHostAlloc(reinterpret_cast<void **>(&link_h), link_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "expert link");
        std::memset(link_h, 0, link_bytes);
        check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&link_d), link_h, 0), "expert link");
        // per-pair rows are for decode steps only (the second half of eh is the host-memory branch's)
        eh = fc::DeviceBuffer(std::size_t(2 * fc::kMaxTokens) * fc::kUsed * fc::kExpertFF * f);
        ypairs = fc::DeviceBuffer(std::size_t(fc::kMaxTokens) * fc::kUsed * fc::kEmbd * f);
        if (cap > fc::kMaxTokens) {
            gpu_sum = buf(fc::kEmbd);
            batch_ws = fc::DeviceBuffer(fc::experts_batch_workspace_bytes(cap));
        }
        // zero-copy reads of cache misses happen in decode steps only (kMaxTokens)
        host_slots = fc::DeviceBuffer(std::size_t(fc::kMaxTokens) * fc::kUsed * sizeof(std::int32_t));
        plan_oncpu = fc::DeviceBuffer(std::size_t(fc::kMaxTokens) * fc::kUsed);
        ypairs_host = fc::DeviceBuffer(std::size_t(fc::kMaxTokens) * fc::kUsed * fc::kEmbd * f);
        ple_hist = fc::DeviceBuffer(std::size_t(fc::kPleHist) * fc::kHcd * f);
        // TODO(merge): the prompt planner lends this VRAM from the expert cache during prompts (kv_stage_borrow);
        // until then deep prompt chunks stage into a pool of their own
        if (kv) {
            const std::int64_t cells = opt.kv_stage_cells < 0 ? opt.max_ctx : opt.kv_stage_cells;
            const std::size_t bytes = kv->stage_bytes(cells);
            if (bytes) kv_stage_own = fc::DeviceBuffer(bytes);
        }
    }

    // VRAM a prompt chunk ending at position pos_end - 1 needs for staging attention K/V (0: none)
    std::size_t kv_stage_bytes(std::int64_t pos_end) const { return kv ? kv->stage_bytes(pos_end) : 0; }
    // TODO(merge): replaced by the prompt planner's arena; a region of >= bytes for the current prompt chunk, or null
    void * kv_stage_borrow(std::size_t bytes) { return bytes && bytes <= kv_stage_own.bytes() ? kv_stage_own.get() : nullptr; }

    void reset() {
        check(cudaStreamSynchronize(stream), "sync");
        for (Layer & L : layers) {
            for (fc::DeviceBuffer * b : {&L.conv_state, &L.S, &L.k_cache, &L.v_cache, &L.idx_raw, &L.blocks})
                if (b->get()) check(cudaMemset(b->get(), 0, b->bytes()), "memset");
        }
        check(cudaMemset(ple_hist.get(), 0, ple_hist.bytes()), "memset");
        history.clear();
        n_past = 0;
        rows_pos0 = 0;
        rows_valid = 0;
        snaps_valid = false;
        if (mtp) {
            mtp->pos = 0;
            check(cudaMemset(mtp->pending_h.get(), 0, mtp->pending_h.bytes()), "memset");
        }
    }

    // recurrent state buffers, in snapshot order
    std::vector<fc::DeviceBuffer *> state_buffers() {
        std::vector<fc::DeviceBuffer *> v;
        for (Layer & L : layers) {
            if (L.recurrent) {
                v.push_back(&L.conv_state);
                v.push_back(&L.S);
            } else {
                v.push_back(&L.idx_raw);  // the ring holds the raw keys of the incomplete QSA block
            }
        }
        v.push_back(&ple_hist);
        if (mtp) {
            v.push_back(&mtp->pending_h);
            // the MTP K/V ring holds only the last positions, which later positions overwrite: it travels with
            // the snapshot (with M.pos, appended after the buffers)
            v.push_back(&mtp->L.k_cache);
            v.push_back(&mtp->L.v_cache);
        }
        return v;
    }

    std::size_t state_bytes() {
        std::size_t bytes = mtp ? sizeof(std::int64_t) : 0;
        for (fc::DeviceBuffer * b : state_buffers()) bytes += b->bytes();
        return bytes;
    }

    EngineSnapshot snapshot() {
        mtp_catchup();
        check(cudaStreamSynchronize(stream), "snapshot");
        EngineSnapshot snap;
        snap.tokens.assign(history.begin(), history.begin() + n_past);
        snap.state.resize(state_bytes());
        std::size_t off = 0;
        for (fc::DeviceBuffer * b : state_buffers()) {
            check(cudaMemcpy(snap.state.data() + off, b->get(), b->bytes(), cudaMemcpyDeviceToHost), "snapshot");
            off += b->bytes();
        }
        if (mtp) std::memcpy(snap.state.data() + off, &mtp->pos, sizeof(std::int64_t));
        return snap;
    }

    void restore(const EngineSnapshot & snap) {
        const std::size_t n = snap.tokens.size();
        if (n > history.size() || !std::equal(snap.tokens.begin(), snap.tokens.end(), history.begin()))
            throw std::runtime_error("engine: the caches no longer hold this snapshot's tokens");
        if (snap.state.size() != state_bytes()) throw std::runtime_error("engine: snapshot from a different model");
        check(cudaStreamSynchronize(stream), "restore");
        std::size_t off = 0;
        for (fc::DeviceBuffer * b : state_buffers()) {
            check(cudaMemcpy(b->get(), snap.state.data() + off, b->bytes(), cudaMemcpyHostToDevice), "restore");
            off += b->bytes();
        }
        n_past = std::int64_t(n);
        rows_pos0 = n_past;
        rows_valid = 0;
        snaps_valid = false;
        if (mtp) {  // pending_h and the K/V ring came back with the snapshot, so did its position
            std::int64_t pos = 0;
            std::memcpy(&pos, snap.state.data() + off, sizeof(std::int64_t));
            mtp->pos = std::min(pos, n_past);
        }
    }

    // ---------------------------------------------------------------------------------------------

    void emit(const char * name, int il, const void * dev, std::int64_t pos, int T, std::int64_t width) {
        if (!hook) return;
        std::vector<float> h(std::size_t(T) * std::size_t(width));
        check(cudaMemcpyAsync(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "emit");
        check(cudaStreamSynchronize(stream), "emit");
        hook(name, il, pos, T, width, h.data());
    }

    void gemv(const fc::DeviceWeight & w, const fc::DeviceBuffer & in, fc::DeviceBuffer & o, int T, std::size_t in_off = 0) {
        fc::gemv(w.view, in.as<float>() + in_off, o.as<float>(), T, stream);
    }

    // several dense layers that read the same input: one fused launch for decode steps
    void linear_multi(std::initializer_list<std::pair<const fc::DeviceWeight *, fc::DeviceBuffer *>> outs, const fc::DeviceBuffer & in, int T) {
        if (T <= fc::kMaxTokens) {
            fc::GemvTarget tg[fc::kMaxMulti];
            int n = 0;
            for (const auto & [w, o] : outs) tg[n++] = {&w->view, o->as<float>()};
            fc::gemv_multi(tg, n, in.as<float>(), T, stream);
        } else {
            for (const auto & [w, o] : outs) gemm->run(w->view, in.as<float>(), o->as<float>(), T);
        }
    }

    // a dense layer over T tokens: matrix-vector kernels for decode steps, FP32 GEMM for prompts
    void linear(const fc::DeviceWeight & w, const fc::DeviceBuffer & in, fc::DeviceBuffer & o, int T) {
        if (T <= fc::kMaxTokens) fc::gemv(w.view, in.as<float>(), o.as<float>(), T, stream);
        else gemm->run(w.view, in.as<float>(), o.as<float>(), T);
    }

    // hyper-connection mixer over T tokens of res starting at token t0
    void hc_mix(const fc::DeviceBuffer & norm, const fc::DeviceWeight & down, const fc::DeviceWeight & up, const fc::DeviceWeight * inj,
                int T, int t0 = 0, const float * src = nullptr) {
        fc::hc_norm((src ? src : res.as<float>()) + std::size_t(t0) * fc::kHcd, norm.as<float>(), xn.as<float>(), T, cfg.rms_eps, stream);
        if (inj) linear_multi({{&down, &lo}, {inj, &inject}}, xn, T);
        else linear(down, xn, lo, T);
        fc::hc_lowrank_act(lo.as<float>(), T * fc::kHcRank, stream);
        linear(up, lo, gate, T);
        fc::hc_gate_mean(xn.as<float>(), gate.as<float>(), mixed.as<float>(), T, stream);
    }

    void ple_rows(std::int64_t pos, std::int32_t * idx) const {
        const int n_gram = cfg.ple_ngram;
        std::int64_t ctx[8];
        ctx[0] = history[std::size_t(pos)];
        bool cut = false;
        for (int s = 1; s < n_gram; ++s) {
            const std::int64_t t = (cut || pos - s < 0) ? -1 : history[std::size_t(pos - s)];
            cut = cut || t < 0 || t == cfg.ple_eos;
            ctx[s] = cut ? cfg.ple_eos : t;
        }
        for (int n = 2; n <= n_gram; ++n) {
            std::uint64_t h = std::uint64_t(ctx[0]) * cfg.ple_multipliers[0];
            for (int j = 1; j < n; ++j) h ^= std::uint64_t(ctx[j]) * cfg.ple_multipliers[std::size_t(j)];
            const int base = (n - 2) * cfg.ple_heads_per_ngram;
            for (int g = 0; g < cfg.ple_heads_per_ngram; ++g) {
                const int head = base + g;
                idx[head] = std::int32_t(h % cfg.ple_vocab[std::size_t(head)] + cfg.ple_offsets[std::size_t(head)]);
            }
        }
    }

    // host part of PLE: the hashed n-gram rows of the step's tokens, dequantized into h_ple
    void ple_host(int T) {
        const std::size_t rb = row_bytes(ple_table->type, fc::kPleHeadDim);
        const std::int64_t n_rows = ple_table->shape[1];
        for (int t = 0; t < T; ++t) {
            std::int32_t rows[fc::kPleHeads];
            ple_rows(n_past + t, rows);
            for (int h = 0; h < fc::kPleHeads; ++h) {
                if (rows[h] < 0 || rows[h] >= n_rows) throw std::runtime_error("engine: PLE row out of range");
                dequantize_row(ple_table->type, ple_table->data + std::size_t(rows[h]) * rb,
                               h_ple->get() + std::size_t(t) * fc::kEmbd + h * fc::kPleHeadDim, fc::kPleHeadDim);
            }
        }
    }

    void run_ple(const Layer & L, int il, int T) {
        if (mtp && T <= fc::kMaxTokens)  // for rollback(): the history before this step
            check(cudaMemcpyAsync(ple_hist_prev.get(), ple_hist.get(), ple_hist.bytes(), cudaMemcpyDeviceToDevice, stream), "ple history");
        check(cudaMemcpyAsync(ple_emb.get(), h_ple->get(), std::size_t(T) * fc::kEmbd * sizeof(float), cudaMemcpyHostToDevice, stream), "ple");
        emit("ple_embd", il, ple_emb.get(), n_past, T, fc::kEmbd);
        linear(L.ple_key, ple_emb, ple_key_out, T);
        linear(L.ple_value, ple_emb, ple_val_out, T);
        fc::ple_gate(ple_key_out.as<float>(), ple_val_out.as<float>(), res.as<float>(), L.ple_nk.as<float>(), L.ple_nq.as<float>(),
                     L.ple_nc.as<float>(), ple_gated.as<float>(), ple_norm.as<float>(), ple_gates.as<float>(), T, cfg.rms_eps, stream);
        emit("ple_gate", il, ple_gates.get(), n_past, T, fc::kHc);
        fc::ple_conv_add(res.as<float>(), ple_gated.as<float>(), ple_norm.as<float>(), L.ple_conv.as<float>(), ple_hist.as<float>(), T, stream);
        emit("ple_out", il, res.get(), n_past, T, fc::kHcd);
    }

    void deltanet(Layer & L, int il, int T) {
        linear_multi({{&L.wqkv, &qkv}, {&L.wgate, &z}, {&L.ssm_beta, &beta}, {&L.ssm_alpha, &alpha}}, mixed, T);
        emit("linear_attn_qkv_mixed", il, qkv.get(), n_past, T, fc::kDnConvDim);
        emit("z", il, z.get(), n_past, T, fc::kDnVDim);
        const bool snap = mtp && T > 1 && T <= fc::kMaxTokens;  // keep per-token states for rollback()
        fc::dn_conv(qkv.as<float>(), L.conv_state.as<float>(), L.conv1d.as<float>(), conv.as<float>(), T, cfg.rms_eps, stream,
                    snap ? L.conv_snap.as<float>() : nullptr);
        fc::dn_recurrence(conv.as<float>(), z.as<float>(), beta.as<float>(), alpha.as<float>(), L.dt.as<float>(), L.a.as<float>(),
                          L.ssm_norm.as<float>(), L.S.as<float>(), dn_out.as<float>(), T, cfg.rms_eps, stream,
                          snap ? L.S_snap.as<float>() : nullptr);
        emit("final_output", il, dn_out.get(), n_past, T, fc::kDnVDim);
        linear(L.ssm_out, dn_out, out, T);
    }

    void attention(Layer & L, int il, int T) {
        const std::int64_t * pos = d_step.as<std::int64_t>();
        linear_multi({{&L.wq, &qfull}, {&L.wk, &k}, {&L.wv, &v}, {&L.idx_k, &kraw}, {&L.idx_q, &qi}}, mixed, T);
        emit("Qcur_full", il, qfull.get(), n_past, T, 2 * fc::kHeads * fc::kHeadDim);
        fc::KvStore kv_store;
        if (L.kv >= 0) {
            kv_store = kv->store(L.kv);
        } else {
            kv_store.k = L.k_cache.as<half>();
            kv_store.v = L.v_cache.as<half>();
        }
        fc::attn_prep(qfull.as<float>(), k.as<float>(), v.as<float>(), L.q_norm.as<float>(), L.k_norm.as<float>(), rope_freq.as<double>(),
                      q.as<float>(), qgate.as<float>(), kv_store, pos, T, cfg.rms_eps, stream);
        emit("Qcur", il, q.get(), n_past, T, fc::kHeads * fc::kHeadDim);
        emit("indexer_k_raw", il, kraw.get(), n_past, T, fc::kIdxDim);
        // QSA: block keys and selections are kept from the first token on, so that past 2051 tokens
        // each query attends to its own 2051 cells; below that the selection is every earlier cell
        fc::store_rows(kraw.as<float>(), L.idx_raw.as<float>(), pos, fc::kIdxDim, T, raw_ring(), stream);
        fc::qsa_update_blocks(L.idx_raw.as<float>(), raw_ring(), L.idx_k_norm.as<float>(), rope_freq.as<double>(), L.blocks.as<float>(), pos,
                              T, cfg.rms_eps, stream);
        fc::qsa_query(qi.as<float>(), L.idx_q_norm.as<float>(), rope_freq.as<double>(), pos, T, cfg.rms_eps, stream);
        fc::qsa_select(qi.as<float>(), L.blocks.as<float>(), pos, T, opt.max_ctx, sel_work.get(), cells.as<std::int32_t>(),
                       n_cells.as<std::int32_t>(), stream);
        if (L.kv >= 0)  // streamed K/V: the page cache or a staging pool (bitwise the same result)
            kv->attend(L.kv, q.as<float>(), qgate.as<float>(), cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), pos, cfg.kq_scale,
                       attn_work.as<float>(), att.as<float>());
        else
            fc::attn_sparse(q.as<float>(), qgate.as<float>(), L.k_cache.as<half>(), L.v_cache.as<half>(), cells.as<std::int32_t>(),
                            n_cells.as<std::int32_t>(), T, cfg.kq_scale, attn_work.as<float>(), att.as<float>(), stream);
        emit("attn_gated", il, att.get(), n_past, T, fc::kHeads * fc::kHeadDim);
        linear(L.wo, att, out, T);
    }

    // CUDA-graph mode: the CPU's share of the experts comes through the mapped ExpertLink of the layer
    void ffn_graph(Layer & L, int il, int T) {
        gemv(L.router, mixed, rlogits, T);
        fc::router_topk(rlogits.as<float>(), ids.as<std::int32_t>(), wts.as<float>(), T, stream);
        const std::int64_t * dseq = d_step.as<std::int64_t>() + 1;
        LayerCache & C = cache[std::size_t(il)];
        const bool zc = !images.empty() && opt.gpu_miss_permille > 0;
        fc::moe_plan(ids.as<std::int32_t>(), C.dmap.as<std::int32_t>(), slots.as<std::int32_t>(), host_slots.as<std::int32_t>(),
                     plan_oncpu.as<std::uint8_t>(), T, zc ? opt.gpu_miss_permille : 0, stream);
        fc::link_signal(ids.as<std::int32_t>(), wts.as<float>(), plan_oncpu.as<std::uint8_t>(), mixed.as<float>(), T, link_d + il, dseq, stream);
        if (zc) {
            // misses read from host memory run beside the cached experts: PCIe and VRAM bandwidth add up
            check(cudaEventRecord(ev_fork, stream), "fork");
            check(cudaStreamWaitEvent(stream2, ev_fork, 0), "fork");
            fc::experts_gpu(C.lay, images_d[std::size_t(il)], host_slots.as<std::int32_t>(), wts.as<float>(), mixed.as<float>(),
                            eh.as<float>() + std::size_t(fc::kMaxTokens) * fc::kUsed * fc::kExpertFF, ypairs_host.as<float>(), T, stream2);
            check(cudaEventRecord(ev_join, stream2), "join");
        }
        fc::experts_gpu(C.lay, C.pool.as<std::uint8_t>(), slots.as<std::int32_t>(), wts.as<float>(), mixed.as<float>(), eh.as<float>(),
                        ypairs.as<float>(), T, stream);
        linear_multi({{&L.sh_gate, &sh_g}, {&L.sh_up, &sh_u}, {&L.sh_gate_inp, &sg}}, mixed, T);
        fc::swiglu(sh_g.as<float>(), sh_u.as<float>(), sh_h.as<float>(), T * fc::kFfShared, stream);
        gemv(L.sh_down, sh_h, sd, T);
        fc::link_wait(link_d + il, dseq, moe.as<float>(), T, d_error.as<int>(), stream);
        if (zc) check(cudaStreamWaitEvent(stream, ev_join, 0), "join");
        fc::moe_combine(ypairs.as<float>(), zc ? ypairs_host.as<float>() : nullptr, moe.as<float>(), sd.as<float>(), sg.as<float>(),
                        out.as<float>(), T, stream);
    }

    // CPU side of a graph-mode step: for every layer, wait for the GPU's selections, compute the
    // experts that are not in VRAM, and hand the sum back.
    void host_experts(int T) {
        const std::size_t ke = std::size_t(T) * fc::kUsed;
        int il = 0;
        try {
            for (; il < cfg.n_layer; ++il) {
                fc::ExpertLink & Lk = link_h[il];
                const auto w0 = clk::now();
                for (unsigned spin = 0; *reinterpret_cast<volatile std::int64_t *>(&Lk.req) != seq; ++spin) {
                    _mm_pause();
                    if ((spin & 0xFFFF) == 0 && clk::now() - w0 > std::chrono::seconds(10))
                        throw std::runtime_error("engine: GPU did not reach layer " + std::to_string(il));
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                const LayerCache & C = cache[std::size_t(il)];
                std::size_t on_cpu = 0, hits = 0;
                for (std::size_t i = 0; i < ke; ++i) {
                    const std::int32_t e = Lk.ids[i];
                    if (e < 0 || e >= fc::kExperts) throw std::runtime_error("engine: router returned an invalid expert");
                    observe(il, e);
                    // the GPU's plan decides which misses it reads from host memory itself
                    h_oncpu->get()[i] = Lk.on_cpu[i] != 0;
                    on_cpu += h_oncpu->get()[i];
                    hits += C.map[std::size_t(e)] >= 0;
                }
                stats.expert_pairs += std::int64_t(ke);
                stats.expert_hits += std::int64_t(hits);
                stats.expert_host_reads += std::int64_t(ke - hits - on_cpu);
                if (on_cpu) {
                    const auto c0 = clk::now();
                    experts->run(il, T, Lk.x, Lk.ids, Lk.weights, h_oncpu->get(), Lk.out);
                    stats.cpu_experts_ms += std::chrono::duration<double, std::milli>(clk::now() - c0).count();
                } else {
                    std::memset(Lk.out, 0, std::size_t(T) * fc::kEmbd * sizeof(float));
                }
                std::atomic_thread_fence(std::memory_order_release);
                *reinterpret_cast<volatile std::int64_t *>(&Lk.done) = seq;
            }
        } catch (...) {
            // release the GPU (it would wait about a second per layer otherwise), then report
            for (; il < cfg.n_layer; ++il) *reinterpret_cast<volatile std::int64_t *>(&link_h[il].done) = seq;
            cudaStreamSynchronize(stream);
            throw;
        }
    }

    void ffn(Layer & L, int il, int T) {
        if (graph_mode) return ffn_graph(L, il, T);
        const std::size_t xe = std::size_t(T) * fc::kEmbd * sizeof(float), ke = std::size_t(T) * fc::kUsed;
        linear(L.router, mixed, rlogits, T);
        emit("ffn_moe_logits", il, rlogits.get(), n_past, T, fc::kExperts);
        fc::router_topk(rlogits.as<float>(), ids.as<std::int32_t>(), wts.as<float>(), T, stream);
        check(cudaMemcpyAsync(h_ids->get(), ids.get(), ke * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream), "ids");
        check(cudaMemcpyAsync(h_w->get(), wts.get(), ke * sizeof(float), cudaMemcpyDeviceToHost, stream), "weights");
        check(cudaMemcpyAsync(h_mixed->get(), mixed.get(), xe, cudaMemcpyDeviceToHost, stream), "ffn input");
        check(cudaEventRecord(ev_router, stream), "event");
        // while the CPU computes the experts that are not cached, the GPU computes the cached ones
        // and the shared expert
        LayerCache & C = cache[std::size_t(il)];
        fc::moe_slots(ids.as<std::int32_t>(), C.dmap.as<std::int32_t>(), slots.as<std::int32_t>(), T, stream);
        const bool batched = T > fc::kMaxTokens;  // prompt chunk: grouped GEMMs per cached expert
        if (batched)
            fc::experts_gpu_batch(C.lay, C.pool.as<std::uint8_t>(), T, mixed.as<float>(), slots.as<std::int32_t>(), wts.as<float>(),
                                  gpu_sum.as<float>(), batch_ws.get(), stream);
        else
            fc::experts_gpu(C.lay, C.pool.as<std::uint8_t>(), slots.as<std::int32_t>(), wts.as<float>(), mixed.as<float>(), eh.as<float>(),
                            ypairs.as<float>(), T, stream);
        linear(L.sh_gate, mixed, sh_g, T);
        linear(L.sh_up, mixed, sh_u, T);
        fc::swiglu(sh_g.as<float>(), sh_u.as<float>(), sh_h.as<float>(), T * fc::kFfShared, stream);
        linear(L.sh_down, sh_h, sd, T);
        linear(L.sh_gate_inp, mixed, sg, T);
        check(cudaEventSynchronize(ev_router), "router");
        if (hook) {
            std::vector<float> fid(ke);
            for (std::size_t i = 0; i < ke; ++i) fid[i] = float(h_ids->get()[i]);
            hook("ffn_moe_topk", il, n_past, T, fc::kUsed, fid.data());
            hook("ffn_moe_weights", il, n_past, T, fc::kUsed, h_w->get());
        }
        std::size_t misses = 0;
        for (std::size_t i = 0; i < ke; ++i) {
            const std::int32_t e = h_ids->get()[i];
            if (e < 0 || e >= fc::kExperts) throw std::runtime_error("engine: router returned an invalid expert");
            observe(il, e);
            h_oncpu->get()[i] = C.map[std::size_t(e)] < 0;
            misses += h_oncpu->get()[i];
        }
        stats.expert_pairs += std::int64_t(ke);
        stats.expert_hits += std::int64_t(ke - misses);
        if (misses) {
            const auto c0 = clk::now();
            if (T <= CpuExperts::kMaxTokens) {
                experts->run(il, T, h_mixed->get(), h_ids->get(), h_w->get(), h_oncpu->get(), h_moe->get());
            } else {
                // prompt chunk: every expert's weights are read once for all of its tokens
                for (int t0 = 0; t0 < T; t0 += CpuExperts::kMaxBatchTokens) {
                    const int n = std::min(CpuExperts::kMaxBatchTokens, T - t0);
                    const std::size_t xo = std::size_t(t0) * fc::kEmbd, ko = std::size_t(t0) * fc::kUsed;
                    experts->run_batch(il, n, h_mixed->get() + xo, h_ids->get() + ko, h_w->get() + ko, h_oncpu->get() + ko,
                                       h_moe->get() + xo);
                }
            }
            stats.cpu_experts_ms += std::chrono::duration<double, std::milli>(clk::now() - c0).count();
            check(cudaMemcpyAsync(moe.get(), h_moe->get(), xe, cudaMemcpyHostToDevice, stream), "moe");
        } else {
            check(cudaMemsetAsync(moe.get(), 0, xe, stream), "moe");
        }
        if (batched) fc::moe_combine_sum(gpu_sum.as<float>(), moe.as<float>(), sd.as<float>(), sg.as<float>(), out.as<float>(), T, stream);
        else fc::moe_combine(ypairs.as<float>(), nullptr, moe.as<float>(), sd.as<float>(), sg.as<float>(), out.as<float>(), T, stream);
    }

    // Host work before a step: token history, embeddings, PLE rows, and the step record.
    void prepare(const std::int32_t * tokens, int T) {
        const std::int64_t pos0 = n_past;
        if (pos0 + T > opt.max_ctx) throw std::runtime_error("engine: context full (max_ctx " + std::to_string(opt.max_ctx) + ")");
        for (int t = 0; t < T; ++t)
            if (tokens[t] < 0 || tokens[t] >= cfg.n_vocab) throw std::runtime_error("engine: token id out of range");
        if (history.size() < std::size_t(pos0 + T)) history.resize(std::size_t(pos0 + T));
        for (int t = 0; t < T; ++t) history[std::size_t(pos0 + t)] = tokens[t];
        const std::size_t rb = row_bytes(tok_embd->type, fc::kEmbd);
        for (int t = 0; t < T; ++t)
            dequantize_row(tok_embd->type, tok_embd->data + std::size_t(tokens[t]) * rb, h_x->get() + std::size_t(t) * fc::kEmbd, fc::kEmbd);
        if (ple_table) ple_host(T);
        h_step.get()[0] = pos0;
        h_step.get()[1] = ++seq;
    }

    // Every GPU operation of a step, in order; host-free in graph mode, so it can be captured.
    // head_rows: how many of the last tokens get logits (at most kMaxTokens)
    void enqueue(int T, int head_rows) {
        const std::int64_t pos0 = n_past;
        check(cudaMemcpyAsync(d_step.get(), h_step.get(), 2 * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream), "step");
        if (kv) kv->begin_step(pos0, T, T > fc::kMaxTokens ? kv_stage_borrow(kv_stage_bytes(pos0 + T)) : nullptr);
        check(cudaMemcpyAsync(x.get(), h_x->get(), std::size_t(T) * fc::kEmbd * sizeof(float), cudaMemcpyHostToDevice, stream), "embed");
        emit("model.input_embed", -1, x.get(), pos0, T, fc::kEmbd);
        fc::hc_expand(x.as<float>(), res.as<float>(), T, stream);

        for (int il = 0; il < cfg.n_layer; ++il) {
            Layer & L = layers[std::size_t(il)];
            if (L.ple) run_ple(L, il, T);
            hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, T);
            emit("hc_attn_mixed", il, mixed.get(), pos0, T, fc::kEmbd);
            if (L.recurrent) deltanet(L, il, T);
            else attention(L, il, T);
            emit(L.recurrent ? "linear_attn_out" : "attn_output", il, out.get(), pos0, T, fc::kEmbd);
            fc::hc_combine(res.as<float>(), out.as<float>(), inject.as<float>(), T, stream);

            hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, T);
            emit("hc_ffn_mixed", il, mixed.get(), pos0, T, fc::kEmbd);
            ffn(L, il, T);
            emit("ffn_out", il, out.get(), pos0, T, fc::kEmbd);
            fc::hc_combine(res.as<float>(), out.as<float>(), inject.as<float>(), T, stream);
            emit("l_out", il, res.get(), pos0, T, fc::kHcd);
        }

        // the head for the last head_rows tokens (all of a graph step: MTP verification reads them all)
        hc_mix(out_hc_norm, out_hc_down, out_hc_up, nullptr, head_rows, T - head_rows);
        gemv(output, mixed, logits, head_rows);
        check(cudaMemcpyAsync(h_logits->get(), logits.get(), std::size_t(head_rows) * cfg.n_vocab * sizeof(float), cudaMemcpyDeviceToHost,
                              stream),
              "logits");
    }

    cudaGraphExec_t graph_for(int T) {
        if (graphs[T]) return graphs[T];
        graph_mode = true;
        cudaGraph_t g = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        try {
            enqueue(T, T);
        } catch (...) {
            cudaStreamEndCapture(stream, &g);
            if (g) cudaGraphDestroy(g);
            graph_mode = false;
            throw;
        }
        check(cudaStreamEndCapture(stream, &g), "capture");
        graph_mode = false;
        const cudaError_t err = cudaGraphInstantiate(&graphs[T], g, 0);
        cudaGraphDestroy(g);
        check(err, "graph instantiate");
        return graphs[T];
    }

    // T tokens (1..cap; all_logits only up to kMaxTokens); returns logits of the last token or of all T
    std::vector<float> step(const std::int32_t * tokens, int T, bool all_logits) {
        mtp_catchup();  // the MTP layer consumes the previous step's hidden rows before res is overwritten
        const std::int64_t pos0 = n_past;
        if (T < 1 || T > cap || (all_logits && T > fc::kMaxTokens)) throw std::runtime_error("engine: bad step size");
        prepare(tokens, T);
        decay(T);
        int head_rows = all_logits ? T : 1;
        if (opt.cuda_graphs && !hook && T <= fc::kMaxTokens) {
            head_rows = T;
            cudaGraphExec_t g = graph_for(T);
            check(cudaGraphLaunch(g, stream), "graph launch");
            cudaStreamQuery(stream);  // submit now (Windows batches work otherwise)
            host_experts(T);
            check(cudaStreamSynchronize(stream), "step");
            int err = 0;
            check(cudaMemcpy(&err, d_error.get(), sizeof(int), cudaMemcpyDeviceToHost), "error flag");
            if (err) throw std::runtime_error("engine: the GPU timed out waiting for the CPU experts");
        } else {
            enqueue(T, head_rows);
            check(cudaStreamSynchronize(stream), "step");
            if (kv && T > fc::kMaxTokens && kv->stats().overflow) throw std::runtime_error("engine: the KV page cache overflowed");
        }
        const int n_out = all_logits ? T : 1, t_first = T - n_out, skip = head_rows - n_out;
        const float * src = h_logits->get() + std::size_t(skip) * cfg.n_vocab;
        std::vector<float> lg(src, src + std::size_t(n_out) * cfg.n_vocab);
        if (hook) {
            emit("result_norm", -1, mixed.as<float>() + std::size_t(skip) * fc::kEmbd, pos0 + t_first, n_out, fc::kEmbd);
            hook("result_output", -1, pos0 + t_first, n_out, cfg.n_vocab, lg.data());
        }
        n_past += T;
        rows_pos0 = pos0;
        rows_valid = T;
        snaps_valid = mtp && T <= fc::kMaxTokens;
        last_T = T;
        return lg;
    }
};

Engine::Engine(const GgufModel & model, EngineOptions options) : impl_(std::make_unique<Impl>(model, options)) {}
Engine::~Engine() = default;

std::vector<float> Engine::forward(const std::vector<std::int32_t> & tokens, bool all_logits) {
    if (tokens.empty()) throw std::runtime_error("engine: forward() needs at least one token");
    const auto t0 = clk::now();
    std::vector<float> all, last;
    const std::size_t n = tokens.size();
    const std::size_t chunk = all_logits ? cuda::kMaxTokens : std::size_t(impl_->cap);
    for (std::size_t i = 0; i < n; i += chunk) {
        const int T = int(std::min(chunk, n - i));
        std::vector<float> lg = impl_->step(tokens.data() + i, T, all_logits);
        if (all_logits) all.insert(all.end(), lg.begin(), lg.end());
        else if (i + T == n) last = std::move(lg);
        ++impl_->stats.steps;
        // a prompt chunk's routing predicts the rest of the prompt: re-rank the cache between chunks
        if (T > cuda::kMaxTokens && i + T < n) impl_->adapt_cache(1 << 30);
    }
    impl_->stats.tokens += std::int64_t(n);
    // re-rank the cached experts after a prompt (its routing predicts the continuation) and from
    // time to time while decoding, sooner while the cache lags the text (under 80% hits)
    Impl & I = *impl_;
    I.tokens_since_adapt += std::int64_t(n);
    const std::int64_t pairs = I.stats.expert_pairs - I.pairs_at_adapt, hits = I.stats.expert_hits - I.hits_at_adapt;
    const bool lagging = pairs > 0 && double(hits) < 0.8 * double(pairs);
    if (n >= 32 || I.tokens_since_adapt >= 256 || (lagging && I.tokens_since_adapt >= 64)) {
        I.adapt_cache(n >= 32 ? (1 << 30) : lagging ? 256 : 128);
        I.tokens_since_adapt = 0;
        I.pairs_at_adapt = I.stats.expert_pairs;
        I.hits_at_adapt = I.stats.expert_hits;
    }
    I.stats.step_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    return all_logits ? all : last;
}

void Engine::reset() { impl_->reset(); }
std::int64_t Engine::n_past() const { return impl_->n_past; }
std::vector<std::int32_t> Engine::tokens() const {
    return std::vector<std::int32_t>(impl_->history.begin(), impl_->history.begin() + impl_->n_past);
}
EngineSnapshot Engine::snapshot() const { return impl_->snapshot(); }
void Engine::restore(const EngineSnapshot & snapshot) { impl_->restore(snapshot); }
int Engine::n_vocab() const { return impl_->cfg.n_vocab; }
void Engine::set_activation_hook(EngineHook hook) { impl_->hook = std::move(hook); }
const EngineStats & Engine::stats() const { return impl_->stats; }
void Engine::save_routing_stats(const std::string & path) const { impl_->save_routing(path); }
bool Engine::has_mtp() const { return impl_->mtp != nullptr; }
KvStreamStats Engine::kv_stream_stats() const { return impl_->kv ? impl_->kv->stats() : KvStreamStats{}; }
std::vector<std::int32_t> Engine::draft(std::int32_t next, int k) { return impl_->draft(next, k); }
void Engine::rollback(int n_keep) { impl_->rollback(n_keep); }

}  // namespace ninfer::flashnext
