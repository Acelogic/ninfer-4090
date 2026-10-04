#include "flashnext/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "flashnext/cpu_experts.h"
#include "flashnext/cuda/gemv.h"
#include "flashnext/cuda/ops.h"
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
        fc::DeviceWeight wq, wk, wv, wo, idx_k;  // attention
        fc::DeviceBuffer q_norm, k_norm;
        fc::DeviceWeight wqkv, wgate, ssm_beta, ssm_alpha, ssm_out;  // Gated DeltaNet
        fc::DeviceBuffer conv1d, dt, a, ssm_norm;
        fc::DeviceWeight ple_key, ple_value;  // PLE
        fc::DeviceBuffer ple_nk, ple_nq, ple_nc, ple_conv;
        fc::DeviceWeight router, sh_gate, sh_up, sh_down, sh_gate_inp;  // FFN
        fc::DeviceBuffer conv_state, S, k_cache, v_cache, idx_raw;      // state
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
    Pinned<float> h_x{fc::kMaxTokens * fc::kEmbd}, h_ple{fc::kMaxTokens * fc::kEmbd}, h_mixed{fc::kMaxTokens * fc::kEmbd},
        h_moe{fc::kMaxTokens * fc::kEmbd}, h_w{fc::kMaxTokens * fc::kUsed};
    Pinned<std::int32_t> h_ids{fc::kMaxTokens * fc::kUsed};

    std::vector<std::int32_t> history;
    std::int64_t n_past = 0;

    Impl(const GgufModel & m, EngineOptions o) : model(m), cfg(ReferenceConfig::from_gguf(m)), opt(o) {
        validate();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        check(cudaEventCreateWithFlags(&ev_router, cudaEventDisableTiming), "event");
        load();
        allocate();
        experts = std::make_unique<CpuExperts>(model, CpuExpertsConfig{opt.cpu_threads, {}});
        reset();
    }
    ~Impl() {
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
        require(cfg.idx_head_dim == fc::kIdxDim, "indexer");
        require(cfg.ple_layer < 0 || (cfg.ple_n_heads == fc::kPleHeads && cfg.ple_head_dim == fc::kPleHeadDim &&
                                      cfg.ple_conv_kernel == fc::kPleKernel && cfg.ple_ngram == fc::kPleDilation),
                "PLE");
        if (opt.max_ctx < 1 || opt.max_ctx > (std::int64_t(1) << 20)) throw std::runtime_error("engine: max_ctx out of range");
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
            } else {
                L.wq = W(b + "attn_q.weight", E, 2 * fc::kHeads * fc::kHeadDim);
                L.wk = W(b + "attn_k.weight", E, fc::kKvHeads * fc::kHeadDim);
                L.wv = W(b + "attn_v.weight", E, fc::kKvHeads * fc::kHeadDim);
                L.wo = W(b + "attn_output.weight", fc::kHeads * fc::kHeadDim, E);
                L.q_norm = V(b + "attn_q_norm.weight", fc::kHeadDim);
                L.k_norm = V(b + "attn_k_norm.weight", fc::kHeadDim);
                L.idx_k = W(b + "indexer.k_proj.weight", E, fc::kIdxDim);
                const std::size_t kv = std::size_t(opt.max_ctx) * fc::kKvHeads * fc::kHeadDim * sizeof(half);
                L.k_cache = fc::DeviceBuffer(kv);
                L.v_cache = fc::DeviceBuffer(kv);
                L.idx_raw = fc::DeviceBuffer(std::size_t(opt.max_ctx) * fc::kIdxDim * sizeof(float));
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
        const std::size_t T = fc::kMaxTokens, f = sizeof(float);
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
        logits = buf(std::size_t(cfg.n_vocab));
        attn_work = fc::DeviceBuffer(fc::attn_work_floats(opt.max_ctx, fc::kMaxTokens) * f);
        ple_hist = fc::DeviceBuffer(std::size_t(fc::kPleHist) * fc::kHcd * f);
    }

    void reset() {
        check(cudaStreamSynchronize(stream), "sync");
        for (Layer & L : layers) {
            for (fc::DeviceBuffer * b : {&L.conv_state, &L.S, &L.k_cache, &L.v_cache, &L.idx_raw})
                if (b->get()) check(cudaMemset(b->get(), 0, b->bytes()), "memset");
        }
        check(cudaMemset(ple_hist.get(), 0, ple_hist.bytes()), "memset");
        history.clear();
        n_past = 0;
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

    // hyper-connection mixer over T tokens of res starting at token t0
    void hc_mix(const fc::DeviceBuffer & norm, const fc::DeviceWeight & down, const fc::DeviceWeight & up, const fc::DeviceWeight * inj,
                int T, int t0 = 0) {
        fc::hc_norm(res.as<float>() + std::size_t(t0) * fc::kHcd, norm.as<float>(), xn.as<float>(), T, cfg.rms_eps, stream);
        gemv(down, xn, lo, T);
        fc::hc_lowrank_act(lo.as<float>(), T * fc::kHcRank, stream);
        gemv(up, lo, gate, T);
        fc::hc_gate_mean(xn.as<float>(), gate.as<float>(), mixed.as<float>(), T, stream);
        if (inj) gemv(*inj, xn, inject, T);
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

    void run_ple(const Layer & L, int il, int T) {
        const std::size_t rb = row_bytes(ple_table->type, fc::kPleHeadDim);
        const std::int64_t n_rows = ple_table->shape[1];
        for (int t = 0; t < T; ++t) {
            std::int32_t rows[fc::kPleHeads];
            ple_rows(n_past + t, rows);
            for (int h = 0; h < fc::kPleHeads; ++h) {
                if (rows[h] < 0 || rows[h] >= n_rows) throw std::runtime_error("engine: PLE row out of range");
                dequantize_row(ple_table->type, ple_table->data + std::size_t(rows[h]) * rb,
                               h_ple.get() + std::size_t(t) * fc::kEmbd + h * fc::kPleHeadDim, fc::kPleHeadDim);
            }
        }
        check(cudaMemcpyAsync(ple_emb.get(), h_ple.get(), std::size_t(T) * fc::kEmbd * sizeof(float), cudaMemcpyHostToDevice, stream), "ple");
        emit("ple_embd", il, ple_emb.get(), n_past, T, fc::kEmbd);
        gemv(L.ple_key, ple_emb, ple_key_out, T);
        gemv(L.ple_value, ple_emb, ple_val_out, T);
        fc::ple_gate(ple_key_out.as<float>(), ple_val_out.as<float>(), res.as<float>(), L.ple_nk.as<float>(), L.ple_nq.as<float>(),
                     L.ple_nc.as<float>(), ple_gated.as<float>(), ple_norm.as<float>(), ple_gates.as<float>(), T, cfg.rms_eps, stream);
        emit("ple_gate", il, ple_gates.get(), n_past, T, fc::kHc);
        fc::ple_conv_add(res.as<float>(), ple_gated.as<float>(), ple_norm.as<float>(), L.ple_conv.as<float>(), ple_hist.as<float>(), T, stream);
        emit("ple_out", il, res.get(), n_past, T, fc::kHcd);
    }

    void deltanet(Layer & L, int il, int T) {
        gemv(L.wqkv, mixed, qkv, T);
        gemv(L.wgate, mixed, z, T);
        gemv(L.ssm_beta, mixed, beta, T);
        gemv(L.ssm_alpha, mixed, alpha, T);
        emit("linear_attn_qkv_mixed", il, qkv.get(), n_past, T, fc::kDnConvDim);
        emit("z", il, z.get(), n_past, T, fc::kDnVDim);
        fc::dn_conv(qkv.as<float>(), L.conv_state.as<float>(), L.conv1d.as<float>(), conv.as<float>(), T, cfg.rms_eps, stream);
        fc::dn_recurrence(conv.as<float>(), z.as<float>(), beta.as<float>(), alpha.as<float>(), L.dt.as<float>(), L.a.as<float>(),
                          L.ssm_norm.as<float>(), L.S.as<float>(), dn_out.as<float>(), T, cfg.rms_eps, stream);
        emit("final_output", il, dn_out.get(), n_past, T, fc::kDnVDim);
        gemv(L.ssm_out, dn_out, out, T);
    }

    void attention(Layer & L, int il, int T) {
        gemv(L.wq, mixed, qfull, T);
        gemv(L.wk, mixed, k, T);
        gemv(L.wv, mixed, v, T);
        emit("Qcur_full", il, qfull.get(), n_past, T, 2 * fc::kHeads * fc::kHeadDim);
        fc::attn_prep(qfull.as<float>(), k.as<float>(), v.as<float>(), L.q_norm.as<float>(), L.k_norm.as<float>(), rope_freq.as<double>(),
                      q.as<float>(), qgate.as<float>(), L.k_cache.as<half>(), L.v_cache.as<half>(), n_past, T, cfg.rms_eps, stream);
        emit("Qcur", il, q.get(), n_past, T, fc::kHeads * fc::kHeadDim);
        gemv(L.idx_k, mixed, kraw, T);
        emit("indexer_k_raw", il, kraw.get(), n_past, T, fc::kIdxDim);
        check(cudaMemcpyAsync(L.idx_raw.as<float>() + n_past * fc::kIdxDim, kraw.get(), std::size_t(T) * fc::kIdxDim * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "indexer keys");
        fc::attn_decode(q.as<float>(), qgate.as<float>(), L.k_cache.as<half>(), L.v_cache.as<half>(), n_past, T, cfg.kq_scale,
                        attn_work.as<float>(), att.as<float>(), stream);
        emit("attn_gated", il, att.get(), n_past, T, fc::kHeads * fc::kHeadDim);
        gemv(L.wo, att, out, T);
    }

    void ffn(Layer & L, int il, int T) {
        const std::size_t xe = std::size_t(T) * fc::kEmbd * sizeof(float), ke = std::size_t(T) * fc::kUsed;
        gemv(L.router, mixed, rlogits, T);
        emit("ffn_moe_logits", il, rlogits.get(), n_past, T, fc::kExperts);
        fc::router_topk(rlogits.as<float>(), ids.as<std::int32_t>(), wts.as<float>(), T, stream);
        check(cudaMemcpyAsync(h_ids.get(), ids.get(), ke * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream), "ids");
        check(cudaMemcpyAsync(h_w.get(), wts.get(), ke * sizeof(float), cudaMemcpyDeviceToHost, stream), "weights");
        check(cudaMemcpyAsync(h_mixed.get(), mixed.get(), xe, cudaMemcpyDeviceToHost, stream), "ffn input");
        check(cudaEventRecord(ev_router, stream), "event");
        // the shared expert runs on the GPU while the CPU computes the routed experts
        gemv(L.sh_gate, mixed, sh_g, T);
        gemv(L.sh_up, mixed, sh_u, T);
        fc::swiglu(sh_g.as<float>(), sh_u.as<float>(), sh_h.as<float>(), T * fc::kFfShared, stream);
        gemv(L.sh_down, sh_h, sd, T);
        gemv(L.sh_gate_inp, mixed, sg, T);
        check(cudaEventSynchronize(ev_router), "router");
        if (hook) {
            std::vector<float> fid(ke);
            for (std::size_t i = 0; i < ke; ++i) fid[i] = float(h_ids.get()[i]);
            hook("ffn_moe_topk", il, n_past, T, fc::kUsed, fid.data());
            hook("ffn_moe_weights", il, n_past, T, fc::kUsed, h_w.get());
        }
        const auto c0 = clk::now();
        experts->run(il, T, h_mixed.get(), h_ids.get(), h_w.get(), nullptr, h_moe.get());
        stats.cpu_experts_ms += std::chrono::duration<double, std::milli>(clk::now() - c0).count();
        check(cudaMemcpyAsync(moe.get(), h_moe.get(), xe, cudaMemcpyHostToDevice, stream), "moe");
        emit("ffn_moe_out", il, moe.get(), n_past, T, fc::kEmbd);
        fc::ffn_combine(moe.as<float>(), sd.as<float>(), sg.as<float>(), out.as<float>(), T, stream);
    }

    // T tokens (1..kMaxTokens); returns logits of the last token or of all T
    std::vector<float> step(const std::int32_t * tokens, int T, bool all_logits) {
        const std::int64_t pos0 = n_past;
        if (pos0 + T > opt.max_ctx) throw std::runtime_error("engine: context full (max_ctx " + std::to_string(opt.max_ctx) + ")");
        // QSA selects a subset of cells only beyond top_k + ratio - 1 tokens; until the sparse path
        // exists, refuse rather than attend densely and give different results.
        if (pos0 + T > std::int64_t(cfg.idx_top_k) + 3) throw std::runtime_error("engine: contexts beyond 2051 tokens need QSA (not implemented yet)");
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || tokens[t] >= cfg.n_vocab) throw std::runtime_error("engine: token id out of range");
            history.push_back(tokens[t]);
        }
        const std::size_t rb = row_bytes(tok_embd->type, fc::kEmbd);
        for (int t = 0; t < T; ++t)
            dequantize_row(tok_embd->type, tok_embd->data + std::size_t(tokens[t]) * rb, h_x.get() + std::size_t(t) * fc::kEmbd, fc::kEmbd);
        check(cudaMemcpyAsync(x.get(), h_x.get(), std::size_t(T) * fc::kEmbd * sizeof(float), cudaMemcpyHostToDevice, stream), "embed");
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

        const int t_first = all_logits ? 0 : T - 1, n_out = T - t_first;
        hc_mix(out_hc_norm, out_hc_down, out_hc_up, nullptr, n_out, t_first);
        emit("result_norm", -1, mixed.get(), pos0 + t_first, n_out, fc::kEmbd);
        gemv(output, mixed, logits, n_out);
        std::vector<float> lg(std::size_t(n_out) * std::size_t(cfg.n_vocab));
        check(cudaMemcpyAsync(lg.data(), logits.get(), lg.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "logits");
        check(cudaStreamSynchronize(stream), "step");
        if (hook) hook("result_output", -1, pos0 + t_first, n_out, cfg.n_vocab, lg.data());
        n_past += T;
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
    for (std::size_t i = 0; i < n; i += cuda::kMaxTokens) {
        const int T = int(std::min<std::size_t>(cuda::kMaxTokens, n - i));
        std::vector<float> lg = impl_->step(tokens.data() + i, T, all_logits);
        if (all_logits) all.insert(all.end(), lg.begin(), lg.end());
        else if (i + T == n) last = std::move(lg);
        ++impl_->stats.steps;
    }
    impl_->stats.tokens += std::int64_t(n);
    impl_->stats.step_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    return all_logits ? all : last;
}

void Engine::reset() { impl_->reset(); }
std::int64_t Engine::n_past() const { return impl_->n_past; }
int Engine::n_vocab() const { return impl_->cfg.n_vocab; }
void Engine::set_activation_hook(EngineHook hook) { impl_->hook = std::move(hook); }
const EngineStats & Engine::stats() const { return impl_->stats; }

}  // namespace ninfer::flashnext
