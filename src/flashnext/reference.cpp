#include "flashnext/reference.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

#include "flashnext/quants.h"

// Every formula below follows the qwen4exp graph of the llama.cpp fork (src/models/qwen4exp.cpp,
// llama-graph.cpp, delta-net-base.cpp, llama-memory-hybrid-idx.cpp) and the CPU semantics of the
// ggml ops it uses (ggml-cpu/ops.cpp). Where ggml fixes a summation order that is cheap to keep,
// the reference keeps it; everything else is plain FP32 with float accumulators (double for the
// RMS/L2 norm sums and softmax denominators, as ggml does).

namespace ninfer::flashnext {

namespace {

// ------------------------------------------------------------------------------------------------
// Thread pool: run(fn) calls fn(thread_index) once on every thread, the caller being thread 0.

class ThreadPool {
public:
    explicit ThreadPool(int n) : n_(std::max(1, n)) {
        for (int i = 1; i < n_; ++i) workers_.emplace_back([this, i] { loop(i); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto & t : workers_) t.join();
    }
    int size() const { return n_; }

    void run(const std::function<void(int)> & fn) {
        if (n_ == 1) {
            fn(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            fn_ = &fn;
            pending_ = n_ - 1;
            error_ = nullptr;
            ++gen_;
        }
        cv_.notify_all();
        std::exception_ptr mine;
        try {
            fn(0);
        } catch (...) {
            mine = std::current_exception();
        }
        std::unique_lock<std::mutex> lock(m_);
        done_.wait(lock, [this] { return pending_ == 0; });
        fn_ = nullptr;
        if (mine) std::rethrow_exception(mine);
        if (error_) std::rethrow_exception(error_);
    }

private:
    void loop(int i) {
        std::uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)> * fn = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                fn = fn_;
            }
            std::exception_ptr err;
            try {
                (*fn)(i);
            } catch (...) {
                err = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(m_);
                if (err && !error_) error_ = err;
                if (--pending_ == 0) done_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)> * fn_ = nullptr;
    int pending_ = 0;
    std::uint64_t gen_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

// fn(item, thread) for item in [0, n), dynamically scheduled.
void parallel_for(ThreadPool & pool, std::int64_t n, const std::function<void(std::int64_t, int)> & fn) {
    if (n <= 0) return;
    if (n == 1 || pool.size() == 1) {
        for (std::int64_t i = 0; i < n; ++i) fn(i, 0);
        return;
    }
    std::atomic<std::int64_t> next{0};
    pool.run([&](int tid) {
        for (;;) {
            const std::int64_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            fn(i, tid);
        }
    });
}

// ------------------------------------------------------------------------------------------------
// Weights: a 2-D view of N rows of K elements in a GGUF tensor (an expert is a slice of a 3-D one).

struct Mat {
    GgufType type{};
    const std::uint8_t * data = nullptr;
    std::int64_t K = 0;  // row length (ggml ne[0])
    std::int64_t N = 0;  // rows
    std::size_t rb = 0;  // bytes per row
    const char * name = "";
};

Mat mat_of(const GgufTensor & t) {
    Mat m;
    m.type = t.type;
    m.data = t.data;
    m.K = t.shape[0];
    m.N = t.elements() / m.K;
    m.rb = row_bytes(t.type, m.K);
    m.name = t.name.c_str();
    return m;
}

Mat expert_of(const GgufTensor & t, std::int64_t e) {
    if (t.shape.size() != 3) throw std::runtime_error(t.name + ": expected a 3-D expert tensor");
    Mat m;
    m.type = t.type;
    m.K = t.shape[0];
    m.N = t.shape[1];
    m.rb = row_bytes(t.type, m.K);
    m.data = t.data + std::size_t(e) * std::size_t(m.N) * m.rb;
    m.name = t.name.c_str();
    return m;
}

std::vector<float> dequant_all(const GgufTensor & t) {
    std::vector<float> v(std::size_t(t.elements()));
    const std::int64_t K = t.shape[0], N = t.elements() / K;
    const std::size_t rb = row_bytes(t.type, K);
    for (std::int64_t r = 0; r < N; ++r) dequantize_row(t.type, t.data + std::size_t(r) * rb, v.data() + r * K, K);
    return v;
}

// ------------------------------------------------------------------------------------------------
// FP32 kernels (AVX-512). Dot products use float lanes; a K tail that is not a multiple of 16 is masked.

inline __mmask16 tail_mask(std::int64_t rem) { return __mmask16((1u << rem) - 1u); }

float dot(const float * a, const float * b, std::int64_t n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    std::int64_t k = 0;
    for (; k + 32 <= n; k += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k + 16), _mm512_loadu_ps(b + k + 16), acc1);
    }
    for (; k + 16 <= n; k += 16) acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), acc0);
    if (k < n) {
        const __mmask16 m = tail_mask(n - k);
        acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + k), _mm512_maskz_loadu_ps(m, b + k), acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// y[t*ldy + r] += sum_k w[r*ldw + k] * x[t*ldx + k] for a 4 x 4 block of rows r and tokens t.
void mk4x4(const float * w, std::int64_t ldw, const float * x, std::int64_t ldx, std::int64_t kk, float * y, std::int64_t ldy) {
    __m512 c00 = _mm512_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
    __m512 c10 = c00, c11 = c00, c12 = c00, c13 = c00;
    __m512 c20 = c00, c21 = c00, c22 = c00, c23 = c00;
    __m512 c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    const float *w0 = w, *w1 = w + ldw, *w2 = w + 2 * ldw, *w3 = w + 3 * ldw;
    const float *x0 = x, *x1 = x + ldx, *x2 = x + 2 * ldx, *x3 = x + 3 * ldx;
    std::int64_t k = 0;
    auto body = [&](__m512 a0, __m512 a1, __m512 a2, __m512 a3, __m512 b0, __m512 b1, __m512 b2, __m512 b3) {
        c00 = _mm512_fmadd_ps(a0, b0, c00); c01 = _mm512_fmadd_ps(a0, b1, c01);
        c02 = _mm512_fmadd_ps(a0, b2, c02); c03 = _mm512_fmadd_ps(a0, b3, c03);
        c10 = _mm512_fmadd_ps(a1, b0, c10); c11 = _mm512_fmadd_ps(a1, b1, c11);
        c12 = _mm512_fmadd_ps(a1, b2, c12); c13 = _mm512_fmadd_ps(a1, b3, c13);
        c20 = _mm512_fmadd_ps(a2, b0, c20); c21 = _mm512_fmadd_ps(a2, b1, c21);
        c22 = _mm512_fmadd_ps(a2, b2, c22); c23 = _mm512_fmadd_ps(a2, b3, c23);
        c30 = _mm512_fmadd_ps(a3, b0, c30); c31 = _mm512_fmadd_ps(a3, b1, c31);
        c32 = _mm512_fmadd_ps(a3, b2, c32); c33 = _mm512_fmadd_ps(a3, b3, c33);
    };
    for (; k + 16 <= kk; k += 16) {
        body(_mm512_loadu_ps(w0 + k), _mm512_loadu_ps(w1 + k), _mm512_loadu_ps(w2 + k), _mm512_loadu_ps(w3 + k),
             _mm512_loadu_ps(x0 + k), _mm512_loadu_ps(x1 + k), _mm512_loadu_ps(x2 + k), _mm512_loadu_ps(x3 + k));
    }
    if (k < kk) {
        const __mmask16 m = tail_mask(kk - k);
        body(_mm512_maskz_loadu_ps(m, w0 + k), _mm512_maskz_loadu_ps(m, w1 + k), _mm512_maskz_loadu_ps(m, w2 + k),
             _mm512_maskz_loadu_ps(m, w3 + k), _mm512_maskz_loadu_ps(m, x0 + k), _mm512_maskz_loadu_ps(m, x1 + k),
             _mm512_maskz_loadu_ps(m, x2 + k), _mm512_maskz_loadu_ps(m, x3 + k));
    }
    float * y0 = y, *y1 = y + ldy, *y2 = y + 2 * ldy, *y3 = y + 3 * ldy;
    y0[0] += _mm512_reduce_add_ps(c00); y1[0] += _mm512_reduce_add_ps(c01);
    y2[0] += _mm512_reduce_add_ps(c02); y3[0] += _mm512_reduce_add_ps(c03);
    y0[1] += _mm512_reduce_add_ps(c10); y1[1] += _mm512_reduce_add_ps(c11);
    y2[1] += _mm512_reduce_add_ps(c12); y3[1] += _mm512_reduce_add_ps(c13);
    y0[2] += _mm512_reduce_add_ps(c20); y1[2] += _mm512_reduce_add_ps(c21);
    y2[2] += _mm512_reduce_add_ps(c22); y3[2] += _mm512_reduce_add_ps(c23);
    y0[3] += _mm512_reduce_add_ps(c30); y1[3] += _mm512_reduce_add_ps(c31);
    y2[3] += _mm512_reduce_add_ps(c32); y3[3] += _mm512_reduce_add_ps(c33);
}

// y[r] += sum_k w[r*ldw + k] * x[k] for 4 rows and one token.
void mk4x1(const float * w, std::int64_t ldw, const float * x, std::int64_t kk, float * y) {
    __m512 c0 = _mm512_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    const float *w0 = w, *w1 = w + ldw, *w2 = w + 2 * ldw, *w3 = w + 3 * ldw;
    std::int64_t k = 0;
    for (; k + 16 <= kk; k += 16) {
        const __m512 b = _mm512_loadu_ps(x + k);
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + k), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + k), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + k), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + k), b, c3);
    }
    if (k < kk) {
        const __mmask16 m = tail_mask(kk - k);
        const __m512 b = _mm512_maskz_loadu_ps(m, x + k);
        c0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w0 + k), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w1 + k), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w2 + k), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w3 + k), b, c3);
    }
    y[0] += _mm512_reduce_add_ps(c0);
    y[1] += _mm512_reduce_add_ps(c1);
    y[2] += _mm512_reduce_add_ps(c2);
    y[3] += _mm512_reduce_add_ps(c3);
}

constexpr std::int64_t kRowBlock = 32;    // rows dequantized together
constexpr std::int64_t kTokBlock = 32;    // tokens per cache tile
constexpr std::int64_t kKBlock = 2048;    // K per cache tile

// y[t*ldy + r] = sum_k w[r*K + k] * x[t*ldx + k], for nn dequantized rows w and tt tokens.
void gemm_block(const float * w, std::int64_t nn, std::int64_t K, const float * x, std::int64_t ldx, std::int64_t tt,
                float * y, std::int64_t ldy) {
    for (std::int64_t t = 0; t < tt; ++t) std::memset(y + t * ldy, 0, std::size_t(nn) * sizeof(float));
    for (std::int64_t k0 = 0; k0 < K; k0 += kKBlock) {
        const std::int64_t kk = std::min(kKBlock, K - k0);
        for (std::int64_t t0 = 0; t0 < tt; t0 += kTokBlock) {
            const std::int64_t t1 = std::min(tt, t0 + kTokBlock);
            std::int64_t r = 0;
            for (; r + 4 <= nn; r += 4) {
                std::int64_t t = t0;
                for (; t + 4 <= t1; t += 4) mk4x4(w + r * K + k0, K, x + t * ldx + k0, ldx, kk, y + t * ldy + r, ldy);
                for (; t < t1; ++t) mk4x1(w + r * K + k0, K, x + t * ldx + k0, kk, y + t * ldy + r);
            }
            for (; r < nn; ++r) {
                for (std::int64_t t = t0; t < t1; ++t) y[t * ldy + r] += dot(w + r * K + k0, x + t * ldx + k0, kk);
            }
        }
    }
}

// Rows [n0, n1) of W times tokens [t0, t1) of X into Y (single thread; wbuf holds kRowBlock*K floats).
void gemm_rows(const Mat & W, std::int64_t n0, std::int64_t n1, const float * X, std::int64_t t0, std::int64_t t1,
               std::int64_t ldx, float * Y, std::int64_t ldy, float * wbuf) {
    for (std::int64_t nb = n0; nb < n1; nb += kRowBlock) {
        const std::int64_t nn = std::min(kRowBlock, n1 - nb);
        for (std::int64_t r = 0; r < nn; ++r) dequantize_row(W.type, W.data + std::size_t(nb + r) * W.rb, wbuf + r * W.K, W.K);
        gemm_block(wbuf, nn, W.K, X + t0 * ldx, ldx, t1 - t0, Y + t0 * ldy + nb, ldy);
    }
}

inline float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float siluf_(float x) { return x / (1.0f + std::exp(-x)); }
inline float softplusf_(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // ggml: x > 20 ? x : log(1 + e^x)

// ggml_rms_norm: double sum of squares, float mean, scale 1/sqrtf(mean + eps); then * w (if any).
void rms_norm(const float * x, float * y, std::int64_t n, float eps, const float * w) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) sum += double(x[i] * x[i]);
    const float mean = float(sum / double(n));
    const float scale = 1.0f / std::sqrt(mean + eps);
    if (w) {
        for (std::int64_t i = 0; i < n; ++i) y[i] = (x[i] * scale) * w[i];
    } else {
        for (std::int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
    }
}

// ggml_l2_norm: x / max(||x||, eps).
void l2_norm(float * x, std::int64_t n, float eps) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) sum += double(x[i] * x[i]);
    const float scale = 1.0f / std::max(float(std::sqrt(sum)), eps);
    for (std::int64_t i = 0; i < n; ++i) x[i] *= scale;
}

// ---- llama.cpp activation rounding, for the emulate_llamacpp diagnostic only ----
enum class ActRound { none, q8_32, q8_32_f16d, q8_256, bf16 };

// q8_1 (CUDA) / q8_0 (CPU, d stored as f16) blocks of 32: d = amax/127, q = round(x/d)
// q8_K (CPU) blocks of 256: iscale = -127/max (signed max), q = round(iscale*x), x' = q/iscale
void round_activations(ActRound mode, const float * x, float * y, std::int64_t n) {
    switch (mode) {
    case ActRound::none:
        std::memcpy(y, x, std::size_t(n) * sizeof(float));
        return;
    case ActRound::bf16:
        for (std::int64_t i = 0; i < n; ++i) {
            std::uint32_t u;
            std::memcpy(&u, &x[i], 4);
            u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
            std::memcpy(&y[i], &u, 4);
        }
        return;
    case ActRound::q8_32:
    case ActRound::q8_32_f16d:
        for (std::int64_t b = 0; b < n; b += 32) {
            const std::int64_t e = std::min(n, b + 32);
            float amax = 0.0f;
            for (std::int64_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(x[i]));
            const float d = amax / 127.0f;
            if (mode == ActRound::q8_32_f16d) {  // ggml-cpu x86 quantize_row_q8_0: q = rne(x * 127/amax), d kept as f16
                const float id = amax != 0.0f ? 127.0f / amax : 0.0f, dq = fp16_to_f32(f32_to_fp16(d));
                for (std::int64_t i = b; i < e; ++i) y[i] = std::nearbyint(x[i] * id) * dq;
            } else {  // CUDA quantize_q8_1: q = round(x / d)
                for (std::int64_t i = b; i < e; ++i) y[i] = amax == 0.0f ? 0.0f : std::round(x[i] / d) * d;
            }
        }
        return;
    case ActRound::q8_256:
        for (std::int64_t b = 0; b < n; b += 256) {
            const std::int64_t e = std::min(n, b + 256);
            float amax = 0.0f, mx = 0.0f;
            for (std::int64_t i = b; i < e; ++i) {
                if (std::fabs(x[i]) > amax) { amax = std::fabs(x[i]); mx = x[i]; }
            }
            if (amax == 0.0f) {
                for (std::int64_t i = b; i < e; ++i) y[i] = 0.0f;
                continue;
            }
            const float iscale = -127.0f / mx;
            for (std::int64_t i = b; i < e; ++i) y[i] = float(std::min(127, int(std::nearbyint(iscale * x[i])))) * (1.0f / iscale);
        }
        return;
    }
}

// what the oracle does to a matmul input: GPU weights (everything but the routed experts) get q8_1
// blocks, cuBLAS rounds the input of BF16 weights to bf16; the CPU experts use the CPU vec_dot type
ActRound gpu_round(GgufType t) {
    switch (t) {
    case GgufType::F32: case GgufType::F16: return ActRound::none;
    case GgufType::BF16: return ActRound::bf16;
    default: return ActRound::q8_32;
    }
}
ActRound cpu_round(GgufType t) {
    switch (t) {
    case GgufType::F32: return ActRound::none;
    case GgufType::BF16: return ActRound::bf16;
    case GgufType::Q8_0: case GgufType::IQ4_NL: return ActRound::q8_32_f16d;
    default: return ActRound::q8_256;
    }
}

float round_f16(float v) { return fp16_to_f32(f32_to_fp16(v)); }

}  // namespace

// ================================================================================================

ReferenceConfig ReferenceConfig::from_gguf(const GgufModel & m) {
    ReferenceConfig c;
    const std::string arch = m.get_string("general.architecture");
    if (arch != "qwen4exp") throw std::runtime_error("reference: expected architecture qwen4exp, got " + arch);
    const std::string p = arch + ".";
    auto key = [&](const char * k) { return p + k; };
    auto int_or = [&](const char * k, std::int64_t def) { return m.has(key(k)) ? m.get_int(key(k)) : def; };

    const int n_all = int(m.get_int(key("block_count")));
    c.n_layer = n_all - int(int_or("nextn_predict_layers", 0));
    c.n_embd = int(m.get_int(key("embedding_length")));
    c.n_vocab = int(m.tensor("token_embd.weight").shape[1]);
    c.rms_eps = float(m.get_float(key("attention.layer_norm_rms_epsilon")));

    c.hc = int(m.get_int(key("hyper_connection.count")));
    c.hc_rank = int(m.get_int(key("hyper_connection.low_rank")));

    c.n_head = int(m.get_int(key("attention.head_count")));
    c.n_head_kv = int(int_or("attention.head_count_kv", c.n_head));
    c.head_dim = int(int_or("attention.key_length", c.n_embd / c.n_head));
    if (int_or("attention.value_length", c.head_dim) != c.head_dim) throw std::runtime_error("reference: key/value head sizes differ");
    c.n_rot = int(int_or("rope.dimension_count", c.head_dim));
    c.rope_base = float(m.has(key("rope.freq_base")) ? m.get_float(key("rope.freq_base")) : 10000.0);
    {
        const auto s = m.get_ints(key("rope.dimension_sections"));
        for (std::size_t i = 0; i < 4 && i < s.size(); ++i) c.rope_sections[i] = int(s[i]);
    }
    // qwen4exp never reads attention.scale: f_attention_scale stays 0, so the scale is 1/sqrt(head_dim)
    c.kq_scale = 1.0f / std::sqrt(float(c.head_dim));

    c.idx_n_head = int(m.get_int(key("attention.indexer.head_count")));
    c.idx_head_dim = int(m.get_int(key("attention.indexer.key_length")));
    c.idx_top_k = int(m.get_int(key("attention.indexer.top_k")));
    c.compress_ratio.assign(std::size_t(n_all), 0);
    if (m.has(key("attention.compress_ratios"))) {
        const auto r = m.get_ints(key("attention.compress_ratios"));
        for (int i = 0; i < n_all; ++i) c.compress_ratio[std::size_t(i)] = int(r.size() == 1 ? r[0] : r[std::size_t(i)]);
    }

    c.ssm_conv = int(m.get_int(key("ssm.conv_kernel")));
    c.ssm_state = int(m.get_int(key("ssm.state_size")));
    c.ssm_k_heads = int(m.get_int(key("ssm.group_count")));
    c.ssm_v_heads = int(m.get_int(key("ssm.time_step_rank")));
    if (m.get_int(key("ssm.inner_size")) != std::int64_t(c.ssm_state) * c.ssm_v_heads) throw std::runtime_error("reference: ssm inner size");

    c.recurrent.assign(std::size_t(n_all), false);
    if (m.has(key("attention.recurrent_layers"))) {
        const auto r = m.get_ints(key("attention.recurrent_layers"));
        for (int i = 0; i < n_all; ++i) c.recurrent[std::size_t(i)] = (r.size() == 1 ? r[0] : r[std::size_t(i)]) != 0;
    } else {
        const int interval = int(int_or("full_attention_interval", 4));
        for (int i = 0; i < n_all; ++i) c.recurrent[std::size_t(i)] = i < c.n_layer && (i + 1) % interval != 0;
    }

    c.n_expert = int(m.get_int(key("expert_count")));
    c.n_expert_used = int(m.get_int(key("expert_used_count")));
    c.n_ff_exp = int(m.get_int(key("expert_feed_forward_length")));
    c.n_ff_shexp = int(int_or("expert_shared_feed_forward_length", c.n_ff_exp));
    c.expert_weights_scale = 0.0f;  // qwen4exp's load_arch_hparams never reads expert_weights_scale

    if (m.has(key("ple.layers"))) {
        const auto layers = m.get_ints(key("ple.layers"));
        if (layers.size() != 1) throw std::runtime_error("reference: exactly one PLE layer is supported");
        c.ple_layer = int(layers[0]);
        c.ple_ngram = int(m.get_int(key("ple.ngram_size")));
        c.ple_heads_per_ngram = int(m.get_int(key("ple.heads_per_ngram")));
        c.ple_n_heads = (c.ple_ngram - 1) * c.ple_heads_per_ngram;
        c.ple_head_dim = int(m.get_int(key("embedding_length_per_layer_input")));
        c.ple_conv_kernel = int(m.get_int(key("ple.conv_kernel")));
        c.ple_eos = m.get_int(key("ple.eos_token_id"));
        c.ple_multipliers = m.get_uints(key("ple.layer_multipliers"));
        c.ple_offsets = m.get_uints(key("ple.head_offsets"));
        c.ple_vocab = m.get_uints(key("ple.head_vocab_sizes"));
        if (int(c.ple_multipliers.size()) < c.ple_ngram || int(c.ple_offsets.size()) < c.ple_n_heads ||
            int(c.ple_vocab.size()) < c.ple_n_heads) {
            throw std::runtime_error("reference: PLE metadata arrays are too short");
        }
        if (c.ple_head_dim * c.ple_n_heads != c.n_embd) throw std::runtime_error("reference: PLE width != n_embd");
        for (int h = 0; h < c.ple_n_heads; ++h) {
            if (c.ple_vocab[std::size_t(h)] == 0) throw std::runtime_error("reference: PLE head vocabulary size 0");
        }
    }
    return c;
}

// ================================================================================================

struct ReferenceModel::Impl {
    const GgufModel & model;
    ReferenceConfig cfg;
    ReferenceOptions opt;
    ThreadPool pool;
    ActivationHook hook;
    ActivationOverride override_fn;

    std::vector<std::vector<float>> wbuf;  // per thread: kRowBlock * max K

    // weights
    Mat tok_embd, output, hc_head_down, hc_head_up;
    std::vector<float> hc_head_norm;
    const GgufTensor * ple_table = nullptr;

    struct Layer {
        bool recurrent = false;
        int ratio = 0;
        Mat hc_attn_down, hc_attn_up, hc_attn_inject, hc_ffn_down, hc_ffn_up, hc_ffn_inject;
        std::vector<float> hc_attn_norm, hc_ffn_norm;
        // attention
        Mat wq, wk, wv, wo, idx_q, idx_k;
        std::vector<float> q_norm, k_norm, idx_q_norm, idx_k_norm;
        // gated deltanet
        Mat wqkv, wgate, ssm_beta, ssm_alpha, ssm_out;
        std::vector<float> ssm_conv1d, ssm_dt, ssm_a, ssm_norm;
        // ple
        bool ple = false;
        Mat ple_key, ple_value;
        std::vector<float> ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
        // moe
        Mat router, sh_gate, sh_up, sh_down;
        std::vector<float> sh_gate_inp;
        const GgufTensor *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    };
    std::vector<Layer> layers;

    // state
    std::int64_t n_past = 0;
    std::vector<std::int32_t> history;  // every token consumed, by position
    struct AttnState {
        std::vector<float> k, v;        // [pos][n_head_kv * head_dim], k normed and rotated
        std::vector<float> idx_raw;     // [pos][idx_head_dim], raw indexer keys
        std::vector<float> idx_blocks;  // [block][idx_head_dim], pooled, normed, rotated (complete blocks)
    };
    struct RecState {
        std::vector<float> conv;  // [ssm_conv - 1][conv_dim], oldest first
        std::vector<float> S;     // [v_heads][state][state]; S[h][j][i], j value index, i key index
    };
    std::vector<AttnState> attn_state;
    std::vector<RecState> rec_state;
    std::vector<float> ple_hist;  // [(kernel - 1) * ngram][hc * n_embd], oldest first

    std::vector<double> inv_freq;  // rope: base^(-2i/n_rot)

    Impl(const GgufModel & m, ReferenceOptions o)
        : model(m), cfg(ReferenceConfig::from_gguf(m)), opt(o),
          pool(o.n_threads > 0 ? o.n_threads : int(std::max(1u, std::thread::hardware_concurrency()))) {
        load();
        reset();
    }

    // a small vector of exactly `want` elements (norm weights, conv taps, biases)
    std::vector<float> vec(const std::string & name, std::int64_t want) const {
        const GgufTensor & t = model.tensor(name);
        if (t.elements() != want) {
            throw std::runtime_error("reference: " + name + " has " + std::to_string(t.elements()) + " elements, want " + std::to_string(want));
        }
        return dequant_all(t);
    }
    Mat W(const std::string & name) const { return mat_of(model.tensor(name)); }

    void load() {
        const int E = cfg.n_embd, HCD = cfg.hc * cfg.n_embd;
        tok_embd = W("token_embd.weight");
        output = model.find("output.weight") ? W("output.weight") : W("token_embd.weight");
        hc_head_norm = vec("output_hc_norm.weight", HCD);
        hc_head_down = W("output_hc_down.weight");
        hc_head_up = W("output_hc_up.weight");
        if (cfg.ple_layer >= 0) {
            ple_table = &model.tensor("per_layer_token_embd.weight");
            if (ple_table->shape.size() < 2 || ple_table->shape[0] != cfg.ple_head_dim) throw std::runtime_error("reference: PLE table shape");
        }

        std::int64_t max_k = 0;
        auto track = [&](const Mat & m) { max_k = std::max(max_k, m.K); return m; };
        auto check = [](const Mat & m, std::int64_t K, std::int64_t N) {
            if (m.K != K || m.N != N) {
                throw std::runtime_error(std::string("reference: unexpected shape for ") + m.name + ": [" +
                                         std::to_string(m.K) + ", " + std::to_string(m.N) + "], want [" +
                                         std::to_string(K) + ", " + std::to_string(N) + "]");
            }
        };
        check(track(hc_head_down), HCD, cfg.hc_rank);
        check(track(hc_head_up), cfg.hc_rank, HCD);
        check(track(output), E, cfg.n_vocab);
        check(tok_embd, E, cfg.n_vocab);

        const int conv_dim = 2 * cfg.ssm_k_heads * cfg.ssm_state + cfg.ssm_v_heads * cfg.ssm_state;
        const int v_dim = cfg.ssm_v_heads * cfg.ssm_state;
        layers.resize(std::size_t(cfg.n_layer));
        for (int il = 0; il < cfg.n_layer; ++il) {
            Layer & L = layers[std::size_t(il)];
            const std::string b = "blk." + std::to_string(il) + ".";
            L.recurrent = cfg.recurrent[std::size_t(il)];
            L.ratio = cfg.compress_ratio[std::size_t(il)];
            L.hc_attn_norm = vec(b + "hc_attn_norm.weight", HCD);
            L.hc_ffn_norm = vec(b + "hc_ffn_norm.weight", HCD);
            check(track(L.hc_attn_down = W(b + "hc_attn_down.weight")), HCD, cfg.hc_rank);
            check(track(L.hc_attn_up = W(b + "hc_attn_up.weight")), cfg.hc_rank, HCD);
            check(track(L.hc_attn_inject = W(b + "hc_attn_inject.weight")), HCD, cfg.hc);
            check(track(L.hc_ffn_down = W(b + "hc_ffn_down.weight")), HCD, cfg.hc_rank);
            check(track(L.hc_ffn_up = W(b + "hc_ffn_up.weight")), cfg.hc_rank, HCD);
            check(track(L.hc_ffn_inject = W(b + "hc_ffn_inject.weight")), HCD, cfg.hc);
            if (!L.recurrent) {
                const int hd = cfg.head_dim;
                check(track(L.wq = W(b + "attn_q.weight")), E, 2 * hd * cfg.n_head);
                check(track(L.wk = W(b + "attn_k.weight")), E, hd * cfg.n_head_kv);
                check(track(L.wv = W(b + "attn_v.weight")), E, hd * cfg.n_head_kv);
                check(track(L.wo = W(b + "attn_output.weight")), hd * cfg.n_head, E);
                L.q_norm = vec(b + "attn_q_norm.weight", hd);
                L.k_norm = vec(b + "attn_k_norm.weight", hd);
                if (L.ratio > 0) {
                    check(track(L.idx_q = W(b + "indexer.q_proj.weight")), E, cfg.idx_n_head * cfg.idx_head_dim);
                    check(track(L.idx_k = W(b + "indexer.k_proj.weight")), E, cfg.idx_head_dim);
                    L.idx_q_norm = vec(b + "indexer.q_norm.weight", cfg.idx_head_dim);
                    L.idx_k_norm = vec(b + "indexer.k_norm.weight", cfg.idx_head_dim);
                }
            } else {
                check(track(L.wqkv = W(b + "attn_qkv.weight")), E, conv_dim);
                check(track(L.wgate = W(b + "attn_gate.weight")), E, v_dim);
                check(track(L.ssm_beta = W(b + "ssm_beta.weight")), E, cfg.ssm_v_heads);
                check(track(L.ssm_alpha = W(b + "ssm_alpha.weight")), E, cfg.ssm_v_heads);
                check(track(L.ssm_out = W(b + "ssm_out.weight")), v_dim, E);
                L.ssm_conv1d = vec(b + "ssm_conv1d.weight", std::int64_t(cfg.ssm_conv) * conv_dim);
                L.ssm_dt = vec(b + "ssm_dt.bias", cfg.ssm_v_heads);
                L.ssm_a = vec(b + "ssm_a", cfg.ssm_v_heads);
                L.ssm_norm = vec(b + "ssm_norm.weight", cfg.ssm_state);
                if (int(L.ssm_conv1d.size()) != cfg.ssm_conv * conv_dim) throw std::runtime_error("reference: ssm_conv1d size");
            }
            if (il == cfg.ple_layer) {
                L.ple = true;
                check(track(L.ple_key = W(b + "ple_key.weight")), E, HCD);
                check(track(L.ple_value = W(b + "ple_value.weight")), E, E);
                L.ple_norm_key = vec(b + "ple_norm_key.weight", HCD);
                L.ple_norm_query = vec(b + "ple_norm_query.weight", HCD);
                L.ple_norm_conv = vec(b + "ple_norm_conv.weight", HCD);
                L.ple_conv1d = vec(b + "ple_conv1d.weight", std::int64_t(cfg.ple_conv_kernel) * HCD);
                if (int(L.ple_conv1d.size()) != cfg.ple_conv_kernel * HCD) throw std::runtime_error("reference: ple_conv1d size");
            }
            check(track(L.router = W(b + "ffn_gate_inp.weight")), E, cfg.n_expert);
            check(track(L.sh_gate = W(b + "ffn_gate_shexp.weight")), E, cfg.n_ff_shexp);
            check(track(L.sh_up = W(b + "ffn_up_shexp.weight")), E, cfg.n_ff_shexp);
            check(track(L.sh_down = W(b + "ffn_down_shexp.weight")), cfg.n_ff_shexp, E);
            L.sh_gate_inp = vec(b + "ffn_gate_inp_shexp.weight", E);
            L.gate_exps = &model.tensor(b + "ffn_gate_exps.weight");
            L.up_exps = &model.tensor(b + "ffn_up_exps.weight");
            L.down_exps = &model.tensor(b + "ffn_down_exps.weight");
            check(track(expert_of(*L.gate_exps, 0)), E, cfg.n_ff_exp);
            check(track(expert_of(*L.up_exps, 0)), E, cfg.n_ff_exp);
            check(track(expert_of(*L.down_exps, 0)), cfg.n_ff_exp, E);
            auto shape_is = [](const GgufTensor & t, std::vector<std::int64_t> want) { return t.shape == want; };
            if (!shape_is(*L.gate_exps, {E, cfg.n_ff_exp, cfg.n_expert}) || !shape_is(*L.up_exps, {E, cfg.n_ff_exp, cfg.n_expert}) ||
                !shape_is(*L.down_exps, {cfg.n_ff_exp, E, cfg.n_expert})) {
                throw std::runtime_error("reference: unexpected routed expert shapes in layer " + std::to_string(il));
            }
        }

        wbuf.assign(std::size_t(pool.size()), std::vector<float>(std::size_t(kRowBlock * max_k)));

        inv_freq.resize(std::size_t(cfg.n_rot / 2));
        for (int i = 0; i < cfg.n_rot / 2; ++i) inv_freq[std::size_t(i)] = std::pow(double(cfg.rope_base), -2.0 * i / cfg.n_rot);
        // imrope with sections [11, 11, 10, 0] gives every one of the 32 rotated pairs a temporal,
        // height or width section, and a text token has the same position in all three, so the
        // rotation is exactly NEOX over the first n_rot dims. Check the assumption about sections.
        const int sect = cfg.rope_sections[0] + cfg.rope_sections[1] + cfg.rope_sections[2] + cfg.rope_sections[3];
        for (int i = 0; i < cfg.n_rot / 2; ++i) {
            const int s = i % sect;
            const bool thw = (s % 3 == 0 && s < 3 * cfg.rope_sections[0]) || (s % 3 == 1 && s < 3 * cfg.rope_sections[1]) ||
                             (s % 3 == 2 && s < 3 * cfg.rope_sections[2]);
            if (!thw) throw std::runtime_error("reference: rope pair uses the extra (e) section; text positions would differ");
        }
    }

    void reset() {
        n_past = 0;
        history.clear();
        const int conv_dim = 2 * cfg.ssm_k_heads * cfg.ssm_state + cfg.ssm_v_heads * cfg.ssm_state;
        attn_state.assign(std::size_t(cfg.n_layer), AttnState{});
        rec_state.assign(std::size_t(cfg.n_layer), RecState{});
        for (int il = 0; il < cfg.n_layer; ++il) {
            if (cfg.recurrent[std::size_t(il)]) {
                rec_state[std::size_t(il)].conv.assign(std::size_t((cfg.ssm_conv - 1) * conv_dim), 0.0f);
                rec_state[std::size_t(il)].S.assign(std::size_t(cfg.ssm_v_heads) * cfg.ssm_state * cfg.ssm_state, 0.0f);
            }
        }
        if (cfg.ple_layer >= 0) {
            ple_hist.assign(std::size_t((cfg.ple_conv_kernel - 1) * cfg.ple_ngram) * cfg.hc * cfg.n_embd, 0.0f);
        }
    }

    void emit(const char * name, int il, std::int64_t T, std::int64_t width, const float * data) {
        if (hook) hook(name, il, n_past, T, width, data);
    }
    // an intermediate that later steps read from `data`: the override may replace it
    void live(const char * name, int il, std::int64_t T, std::int64_t width, float * data) {
        emit(name, il, T, width, data);
        if (override_fn) override_fn(name, il, n_past, T, width, data);
    }

    // ---------------------------------------------------------------------------- matmul
    // Y[t*ldy + n] = W[n] . X[t*ldx]  for every row n of W and t < T.
    void matmul(const Mat & w, const float * X, std::int64_t T, std::int64_t ldx, float * Y, std::int64_t ldy) {
        std::vector<float> rounded;
        const ActRound mode = opt.emulate == ReferenceOptions::Emulate::llamacpp_cuda ? gpu_round(w.type)
                              : opt.emulate == ReferenceOptions::Emulate::llamacpp_cpu ? cpu_round(w.type)
                                                                                       : ActRound::none;
        if (mode != ActRound::none) {
            rounded.resize(std::size_t(T * w.K));
            for (std::int64_t t = 0; t < T; ++t) round_activations(mode, X + t * ldx, rounded.data() + t * w.K, w.K);
            X = rounded.data();
            ldx = w.K;
        }
        const std::int64_t n_rc = (w.N + kRowBlock - 1) / kRowBlock;
        std::int64_t n_tc = 1;
        const std::int64_t want = 4 * pool.size();
        if (n_rc < want && T > 64) n_tc = std::min((T + 63) / 64, (want + n_rc - 1) / n_rc);
        const std::int64_t tc = (T + n_tc - 1) / n_tc;
        parallel_for(pool, n_rc * n_tc, [&](std::int64_t item, int tid) {
            const std::int64_t rc = item % n_rc, ti = item / n_rc;
            const std::int64_t n0 = rc * kRowBlock, n1 = std::min(w.N, n0 + kRowBlock);
            const std::int64_t t0 = ti * tc, t1 = std::min(T, t0 + tc);
            if (t0 >= t1) return;
            gemm_rows(w, n0, n1, X, t0, t1, ldx, Y, ldy, wbuf[std::size_t(tid)].data());
        });
    }
    std::vector<float> matmul(const Mat & w, const std::vector<float> & X, std::int64_t T) {
        std::vector<float> Y(std::size_t(T * w.N));
        matmul(w, X.data(), T, w.K, Y.data(), w.N);
        return Y;
    }

    // ---------------------------------------------------------------------------- RoPE
    // NEOX rotation of the first n_rot dims: pairs (i, i + n_rot/2), angle pos * base^(-2i/n_rot).
    void rope(float * x, std::int64_t pos) const {
        const int half = cfg.n_rot / 2;
        for (int i = 0; i < half; ++i) {
            const double theta = double(pos) * inv_freq[std::size_t(i)];
            const float c = float(std::cos(theta)), s = float(std::sin(theta));
            const float x0 = x[i], x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }

    // ---------------------------------------------------------------------------- hyper-connections
    // build_hc_mix: per-stream RMSNorm with the [hc*E] gamma, a low-rank sigmoid gate, then the mean
    // of the gated streams. inject (if wanted) gets the [hc] scatter logits.
    void hc_mix(int il, const std::vector<float> & res, std::int64_t T, const std::vector<float> & norm, const Mat & down,
                const Mat & up, const Mat * inject_w, std::vector<float> & mixed, std::vector<float> * inject) {
        const int E = cfg.n_embd, hc = cfg.hc, HCD = hc * E;
        std::vector<float> xn(std::size_t(T * HCD));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            for (int c = 0; c < hc; ++c) {
                rms_norm(res.data() + t * HCD + c * E, xn.data() + t * HCD + c * E, E, cfg.rms_eps, norm.data() + c * E);
            }
        });
        live("hc_norm", il, T, HCD, xn.data());
        std::vector<float> lo = matmul(down, xn, T);
        const float inv_hc = 1.0f / float(hc);
        for (auto & v : lo) v = siluf_(v * inv_hc);
        std::vector<float> gate = matmul(up, lo, T);
        if (hook) {
            std::vector<float> sg(gate.size());
            for (std::size_t i = 0; i < gate.size(); ++i) sg[i] = sigmoidf_(gate[i]);
            emit("hc_gate", il, T, HCD, sg.data());
        }
        mixed.assign(std::size_t(T * E), 0.0f);
        parallel_for(pool, T, [&](std::int64_t t, int) {
            const float * x = xn.data() + t * HCD;
            const float * g = gate.data() + t * HCD;
            float * m = mixed.data() + t * E;
            for (int e = 0; e < E; ++e) m[e] = x[e] * sigmoidf_(g[e]);
            for (int c = 1; c < hc; ++c) {
                for (int e = 0; e < E; ++e) m[e] = m[e] + x[c * E + e] * sigmoidf_(g[c * E + e]);
            }
            for (int e = 0; e < E; ++e) m[e] *= inv_hc;
        });
        if (inject) {
            *inject = matmul(*inject_w, xn, T);
            live("hc_inject", il, T, hc, inject->data());
        }
    }

    // build_hc_combine: res[c] += block_out * 2*sigmoid(inject[c]/hc).
    void hc_combine(int il, std::vector<float> & res, std::int64_t T, const std::vector<float> & out, const std::vector<float> & inject) {
        const int E = cfg.n_embd, hc = cfg.hc;
        const float inv_hc = 1.0f / float(hc);
        parallel_for(pool, T, [&](std::int64_t t, int) {
            for (int c = 0; c < hc; ++c) {
                const float w = 2.0f * sigmoidf_(inject[std::size_t(t * hc + c)] * inv_hc);
                float * r = res.data() + (t * hc + c) * E;
                const float * o = out.data() + t * E;
                for (int e = 0; e < E; ++e) r[e] = r[e] + o[e] * w;
            }
        });
        live("hc_combine", il, T, std::int64_t(hc) * E, res.data());
    }

    // ---------------------------------------------------------------------------- PLE
    // Row indices of the n-gram hash heads for the token at `pos` (llm_graph_input_ple::set_input).
    void ple_rows(std::int64_t pos, std::int32_t * idx) const {
        const int n_gram = cfg.ple_ngram;
        std::vector<std::int64_t> ctx(static_cast<std::size_t>(n_gram));
        ctx[0] = history[std::size_t(pos)];
        bool cut = false;
        for (int s = 1; s < n_gram; ++s) {
            // a predecessor before the sequence start reads as EOS; an EOS cuts everything at or before it
            const std::int64_t t = (cut || pos - s < 0) ? -1 : history[std::size_t(pos - s)];
            cut = cut || t < 0 || t == cfg.ple_eos;
            ctx[std::size_t(s)] = cut ? cfg.ple_eos : t;
        }
        for (int n = 2; n <= n_gram; ++n) {
            std::uint64_t mixed = std::uint64_t(ctx[0]) * cfg.ple_multipliers[0];
            for (int j = 1; j < n; ++j) mixed ^= std::uint64_t(ctx[std::size_t(j)]) * cfg.ple_multipliers[std::size_t(j)];
            const int base = (n - 2) * cfg.ple_heads_per_ngram;
            for (int g = 0; g < cfg.ple_heads_per_ngram; ++g) {
                const int h = base + g;
                idx[h] = std::int32_t(mixed % cfg.ple_vocab[std::size_t(h)] + cfg.ple_offsets[std::size_t(h)]);
            }
        }
    }

    void ple(const Layer & L, int il, std::vector<float> & res, std::int64_t T) {
        const int E = cfg.n_embd, hc = cfg.hc, HCD = hc * E, nh = cfg.ple_n_heads, hd = cfg.ple_head_dim;
        std::vector<std::int32_t> rows(std::size_t(T * nh));
        std::vector<float> emb(std::size_t(T * E));
        const std::size_t rb = row_bytes(ple_table->type, hd);
        const std::int64_t n_rows = ple_table->shape[1];
        for (std::int64_t t = 0; t < T; ++t) ple_rows(n_past + t, rows.data() + t * nh);
        parallel_for(pool, T, [&](std::int64_t t, int) {
            for (int h = 0; h < nh; ++h) {
                const std::int64_t r = rows[std::size_t(t * nh + h)];
                if (r < 0 || r >= n_rows) throw std::runtime_error("reference: PLE row out of range");
                // get_rows lays the heads out slowest: emb[h*hd + d]
                dequantize_row(ple_table->type, ple_table->data + std::size_t(r) * rb, emb.data() + t * E + h * hd, hd);
            }
        });
        live("ple_embd", il, T, E, emb.data());
        std::vector<float> key = matmul(L.ple_key, emb, T);      // [T][hc*E]
        std::vector<float> value = matmul(L.ple_value, emb, T);  // [T][E]

        std::vector<float> gated(std::size_t(T * HCD)), normalized(std::size_t(T * HCD)), gates(std::size_t(T * hc));
        const float inv_sqrt_e = 1.0f / std::sqrt(float(E));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            std::vector<float> kn(static_cast<std::size_t>(E)), qn(static_cast<std::size_t>(E));
            for (int c = 0; c < hc; ++c) {
                rms_norm(key.data() + t * HCD + c * E, kn.data(), E, cfg.rms_eps, L.ple_norm_key.data() + c * E);
                rms_norm(res.data() + t * HCD + c * E, qn.data(), E, cfg.rms_eps, L.ple_norm_query.data() + c * E);
                double sum = 0.0;
                for (int e = 0; e < E; ++e) sum += double(kn[std::size_t(e)] * qn[std::size_t(e)]);
                const float s = float(sum) * inv_sqrt_e;
                // signed square root, clamped away from zero, then a sigmoid
                const float mag = std::sqrt(std::min(std::max(std::fabs(s), 1e-6f), INFINITY));
                const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
                const float g = sigmoidf_(sgn * mag);
                gates[std::size_t(t * hc + c)] = g;
                for (int e = 0; e < E; ++e) gated[std::size_t(t * HCD + c * E + e)] = value[std::size_t(t * E + e)] * g;
            }
            for (int c = 0; c < hc; ++c) {
                rms_norm(gated.data() + t * HCD + c * E, normalized.data() + t * HCD + c * E, E, cfg.rms_eps,
                         L.ple_norm_conv.data() + c * E);
            }
        });
        emit("ple_gate", il, T, hc, gates.data());

        // depthwise causal conv over time, kernel K, dilation n_gram:
        //   out[ch, t] = sum_k w[ch, k] * x[ch, t - (K-1-k)*dil], history of earlier tokens prepended
        const int K = cfg.ple_conv_kernel, dil = cfg.ple_ngram, hist = (K - 1) * dil;
        std::vector<float> padded(std::size_t((hist + T) * HCD));
        std::memcpy(padded.data(), ple_hist.data(), ple_hist.size() * sizeof(float));
        std::memcpy(padded.data() + std::size_t(hist) * HCD, normalized.data(), normalized.size() * sizeof(float));
        std::vector<float> conv_out(std::size_t(T * HCD));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            for (int ch = 0; ch < HCD; ++ch) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float term = padded[std::size_t((hist + t - (K - 1 - k) * dil) * HCD + ch)] * L.ple_conv1d[std::size_t(ch * K + k)];
                    acc = k == 0 ? term : acc + term;
                }
                conv_out[std::size_t(t * HCD + ch)] = siluf_(acc);
            }
        });
        std::memcpy(ple_hist.data(), padded.data() + std::size_t(T) * HCD, ple_hist.size() * sizeof(float));
        emit("ple_conv_out", il, T, HCD, conv_out.data());

        // hidden + (gated + conv)
        for (std::size_t i = 0; i < res.size(); ++i) res[i] = res[i] + (gated[i] + conv_out[i]);
        live("ple_out", il, T, HCD, res.data());
    }

    // ---------------------------------------------------------------------------- Gated DeltaNet
    std::vector<float> deltanet(const Layer & L, int il, const std::vector<float> & x, std::int64_t T) {
        const int Sd = cfg.ssm_state, Hk = cfg.ssm_k_heads, Hv = cfg.ssm_v_heads;
        const int key_dim = Hk * Sd, v_dim = Hv * Sd, conv_dim = 2 * key_dim + v_dim, KC = cfg.ssm_conv;
        RecState & st = rec_state[std::size_t(il)];

        std::vector<float> qkv = matmul(L.wqkv, x, T);       // [T][conv_dim]: q | k | v
        std::vector<float> z = matmul(L.wgate, x, T);        // [T][v_dim]
        std::vector<float> beta = matmul(L.ssm_beta, x, T);  // [T][Hv]
        std::vector<float> alpha = matmul(L.ssm_alpha, x, T);
        live("linear_attn_qkv_mixed", il, T, conv_dim, qkv.data());
        live("z", il, T, v_dim, z.data());

        // causal conv (kernel KC) over [state | inputs], then SiLU; ggml_ssm_conv sums taps oldest first
        std::vector<float> conv(std::size_t(T * conv_dim));
        const int ns = KC - 1;
        parallel_for(pool, (conv_dim + 255) / 256, [&](std::int64_t blk, int) {
            const int c0 = int(blk) * 256, c1 = std::min(conv_dim, c0 + 256);
            std::vector<float> win(std::size_t(ns + T));
            for (int c = c0; c < c1; ++c) {
                for (int s = 0; s < ns; ++s) win[std::size_t(s)] = st.conv[std::size_t(s * conv_dim + c)];
                for (std::int64_t t = 0; t < T; ++t) win[std::size_t(ns + t)] = qkv[std::size_t(t * conv_dim + c)];
                const float * w = L.ssm_conv1d.data() + std::size_t(c) * KC;
                for (std::int64_t t = 0; t < T; ++t) {
                    float sum = 0.0f;
                    for (int i = 0; i < KC; ++i) sum += win[std::size_t(t + i)] * w[i];
                    conv[std::size_t(t * conv_dim + c)] = siluf_(sum);
                }
                for (int s = 0; s < ns; ++s) st.conv[std::size_t(s * conv_dim + c)] = win[std::size_t(T + s)];
            }
        });
        live("conv_output_silu", il, T, conv_dim, conv.data());

        // L2-normalised q and k per head; beta = sigmoid; g = softplus(alpha + dt_bias) * ssm_a
        parallel_for(pool, T, [&](std::int64_t t, int) {
            float * row = conv.data() + t * conv_dim;
            for (int h = 0; h < 2 * Hk; ++h) l2_norm(row + h * Sd, Sd, cfg.rms_eps);
            for (int h = 0; h < Hv; ++h) {
                float & b = beta[std::size_t(t * Hv + h)];
                b = sigmoidf_(b);
                float & a = alpha[std::size_t(t * Hv + h)];
                a = softplusf_(a + L.ssm_dt[std::size_t(h)]) * L.ssm_a[std::size_t(h)];
            }
        });

        // gated delta rule, one v head per work item; v head h reads q/k head h % Hk (ggml tiling)
        std::vector<float> o(std::size_t(T * v_dim));
        const float scale = 1.0f / std::sqrt(float(Sd));
        parallel_for(pool, Hv, [&](std::int64_t h, int) {
            float * S = st.S.data() + std::size_t(h) * Sd * Sd;
            const int hk = int(h % Hk);
            for (std::int64_t t = 0; t < T; ++t) {
                const float * row = conv.data() + t * conv_dim;
                const float * q = row + hk * Sd;
                const float * k = row + key_dim + hk * Sd;
                const float * v = row + 2 * key_dim + h * Sd;
                const float decay = std::exp(alpha[std::size_t(t * Hv + h)]);
                const float b = beta[std::size_t(t * Hv + h)];
                float * out = o.data() + t * v_dim + h * Sd;
                const __m512 vdecay = _mm512_set1_ps(decay);
                for (int j = 0; j < Sd; ++j) {
                    float * Sj = S + std::size_t(j) * Sd;
                    for (int i = 0; i < Sd; i += 16) _mm512_storeu_ps(Sj + i, _mm512_mul_ps(_mm512_loadu_ps(Sj + i), vdecay));
                    const float d = (v[j] - dot(Sj, k, Sd)) * b;
                    const __m512 vd = _mm512_set1_ps(d);
                    for (int i = 0; i < Sd; i += 16) {
                        _mm512_storeu_ps(Sj + i, _mm512_fmadd_ps(_mm512_loadu_ps(k + i), vd, _mm512_loadu_ps(Sj + i)));
                    }
                    out[j] = dot(Sj, q, Sd) * scale;
                }
            }
        });
        live("attn_output", il, T, v_dim, o.data());

        // gated RMSNorm: rms_norm(o) * ssm_norm * sigmoid(z)  (sigmoid, not Qwen3.5's silu)
        std::vector<float> gn(std::size_t(T * v_dim));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            for (int h = 0; h < Hv; ++h) {
                const std::size_t off = std::size_t(t * v_dim + h * Sd);
                rms_norm(o.data() + off, gn.data() + off, Sd, cfg.rms_eps, L.ssm_norm.data());
                for (int d = 0; d < Sd; ++d) gn[off + d] = gn[off + d] * sigmoidf_(z[off + d]);
            }
        });
        live("final_output", il, T, v_dim, gn.data());
        return matmul(L.ssm_out, gn, T);
    }

    // ---------------------------------------------------------------------------- attention + QSA
    std::vector<float> attention(const Layer & L, int il, const std::vector<float> & x, std::int64_t T) {
        const int hd = cfg.head_dim, nh = cfg.n_head, nkv = cfg.n_head_kv;
        const int kvw = nkv * hd;
        AttnState & st = attn_state[std::size_t(il)];

        std::vector<float> qf = matmul(L.wq, x, T);  // [T][nh * 2*hd]: per head [q | gate]
        std::vector<float> kc = matmul(L.wk, x, T);  // [T][kvw]
        std::vector<float> vc = matmul(L.wv, x, T);
        live("Qcur_full", il, T, std::int64_t(nh) * 2 * hd, qf.data());

        std::vector<float> q(std::size_t(T * nh * hd)), gate(std::size_t(T * nh * hd));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            const std::int64_t pos = n_past + t;
            for (int h = 0; h < nh; ++h) {
                const float * src = qf.data() + t * nh * 2 * hd + std::int64_t(h) * 2 * hd;
                float * dq = q.data() + (t * nh + h) * hd;
                rms_norm(src, dq, hd, cfg.rms_eps, L.q_norm.data());
                rope(dq, pos);
                std::memcpy(gate.data() + (t * nh + h) * hd, src + hd, std::size_t(hd) * sizeof(float));
            }
            for (int h = 0; h < nkv; ++h) {
                float * dk = kc.data() + t * kvw + h * hd;
                rms_norm(dk, dk, hd, cfg.rms_eps, L.k_norm.data());
                rope(dk, pos);
            }
        });
        live("Qcur", il, T, std::int64_t(nh) * hd, q.data());
        live("Kcur", il, T, kvw, kc.data());
        live("Vcur", il, T, kvw, vc.data());
        if (opt.emulate != ReferenceOptions::Emulate::none) {  // f16 KV cache, f16 queries
            for (auto & v : kc) v = round_f16(v);
            for (auto & v : vc) v = round_f16(v);
            for (auto & v : q) v = round_f16(v);
        }
        st.k.insert(st.k.end(), kc.begin(), kc.end());
        st.v.insert(st.v.end(), vc.begin(), vc.end());

        // QSA (build_qsa_top_k): select the cells each query attends to. Empty = dense.
        std::vector<std::vector<std::int32_t>> sel(static_cast<std::size_t>(T));
        if (L.ratio > 0) qsa_select(L, il, x, T, sel);

        // softmax(q k^T * scale) v over the visible (and, for QSA, selected) cells; GQA head h -> h / (nh/nkv)
        std::vector<float> att(std::size_t(T * nh * hd));
        const int group = nh / nkv;
        parallel_for(pool, T * nh, [&](std::int64_t item, int) {
            const std::int64_t t = item / nh;
            const int h = int(item % nh), hk = h / group;
            const std::int64_t pos = n_past + t;
            const float * qh = q.data() + (t * nh + h) * hd;
            const std::vector<std::int32_t> & cells = sel[std::size_t(t)];
            const std::int64_t n = cells.empty() ? pos + 1 : std::int64_t(cells.size());
            std::vector<float> s(static_cast<std::size_t>(n));
            float mx = -INFINITY;
            for (std::int64_t j = 0; j < n; ++j) {
                const std::int64_t p = cells.empty() ? j : cells[std::size_t(j)];
                s[std::size_t(j)] = dot(qh, st.k.data() + p * kvw + hk * hd, hd) * cfg.kq_scale;
                mx = std::max(mx, s[std::size_t(j)]);
            }
            double sum = 0.0;
            for (std::int64_t j = 0; j < n; ++j) {
                s[std::size_t(j)] = std::exp(s[std::size_t(j)] - mx);
                sum += s[std::size_t(j)];
            }
            float inv = float(1.0 / sum);
            if (opt.emulate == ReferenceOptions::Emulate::llamacpp_cpu) {  // softmax, then f16 probabilities into V
                for (std::int64_t j = 0; j < n; ++j) s[std::size_t(j)] = round_f16(s[std::size_t(j)] * inv);
                inv = 1.0f;
            } else if (opt.emulate == ReferenceOptions::Emulate::llamacpp_cuda) {  // f16 unnormalised probabilities
                for (std::int64_t j = 0; j < n; ++j) s[std::size_t(j)] = round_f16(s[std::size_t(j)]);
            }
            float * out = att.data() + (t * nh + h) * hd;
            std::vector<float> acc(std::size_t(hd), 0.0f);
            for (std::int64_t j = 0; j < n; ++j) {
                const std::int64_t p = cells.empty() ? j : cells[std::size_t(j)];
                const float * vh = st.v.data() + p * kvw + hk * hd;
                const __m512 pj = _mm512_set1_ps(s[std::size_t(j)]);
                for (int d = 0; d < hd; d += 16) {
                    _mm512_storeu_ps(acc.data() + d, _mm512_fmadd_ps(pj, _mm512_loadu_ps(vh + d), _mm512_loadu_ps(acc.data() + d)));
                }
            }
            const float * g = gate.data() + (t * nh + h) * hd;
            for (int d = 0; d < hd; ++d) out[d] = (acc[std::size_t(d)] * inv) * sigmoidf_(g[d]);
        });
        live("attn_gated", il, T, std::int64_t(nh) * hd, att.data());
        return matmul(L.wo, att, T);
    }

    // Indexer keys are cached raw; a block of `ratio` consecutive positions is pooled (mean), then
    // RMS-normed and rotated at the block's first position. A query at pos attends to:
    //   - every visible cell, when pos + 1 <= top_k + ratio - 1 (the top-k width covers them all);
    //   - otherwise the tail (positions from the start of its own incomplete block up to pos, which
    //     the bias makes always visible) plus the best (width - tail) cells of the complete blocks,
    //     ranked by the block score sum_h relu(q_h . k_block). Ties are broken toward lower positions.
    void qsa_select(const Layer & L, int il, const std::vector<float> & x, std::int64_t T,
                    std::vector<std::vector<std::int32_t>> & sel) {
        AttnState & st = attn_state[std::size_t(il)];
        const int r = L.ratio, dim = cfg.idx_head_dim, nih = cfg.idx_n_head;
        std::vector<float> kraw = matmul(L.idx_k, x, T);  // [T][dim]
        live("indexer_k_raw", il, T, dim, kraw.data());
        st.idx_raw.insert(st.idx_raw.end(), kraw.begin(), kraw.end());

        // complete blocks
        const std::int64_t n_tok = n_past + T;
        const std::int64_t have = std::int64_t(st.idx_blocks.size()) / dim, full = n_tok / r;
        if (full > have) {
            st.idx_blocks.resize(std::size_t(full * dim));
            for (std::int64_t b = have; b < full; ++b) {
                std::vector<float> pooled(static_cast<std::size_t>(dim));
                for (int d = 0; d < dim; ++d) {
                    float acc = st.idx_raw[std::size_t((b * r) * dim + d)];
                    for (int i = 1; i < r; ++i) acc = acc + st.idx_raw[std::size_t((b * r + i) * dim + d)];
                    pooled[std::size_t(d)] = acc * (1.0f / float(r));
                }
                float * kb = st.idx_blocks.data() + b * dim;
                rms_norm(pooled.data(), kb, dim, cfg.rms_eps, L.idx_k_norm.data());
                rope(kb, b * r);
            }
        }
        if (full > 0) emit("indexer_blocks", il, full, dim, st.idx_blocks.data());  // every complete block, [block][dim]

        const std::int64_t width = std::int64_t(cfg.idx_top_k) + r - 1;
        bool any = false;
        for (std::int64_t t = 0; t < T; ++t) any = any || (n_past + t + 1 > width);
        if (!any) return;

        std::vector<float> qi = matmul(L.idx_q, x, T);  // [T][nih * dim]
        emit("indexer_q_proj", il, T, std::int64_t(nih) * dim, qi.data());
        parallel_for(pool, T, [&](std::int64_t t, int) {
            const std::int64_t pos = n_past + t;
            float * qt = qi.data() + t * nih * dim;
            for (int h = 0; h < nih; ++h) {
                rms_norm(qt + h * dim, qt + h * dim, dim, cfg.rms_eps, L.idx_q_norm.data());
                rope(qt + h * dim, pos);
            }
            if (pos + 1 <= width) return;  // dense
            const std::int64_t tail_start = (pos + 1) / r * r;
            const std::int64_t n_blocks = tail_start / r;  // complete blocks, all visible
            std::vector<std::pair<float, std::int32_t>> score(static_cast<std::size_t>(n_blocks));
            for (std::int64_t b = 0; b < n_blocks; ++b) {
                const float * kb = st.idx_blocks.data() + b * dim;
                float s = 0.0f;
                for (int h = 0; h < nih; ++h) s = s + std::max(0.0f, dot(qt + h * dim, kb, dim));
                score[std::size_t(b)] = {s, std::int32_t(b)};
            }
            std::sort(score.begin(), score.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            std::vector<std::int32_t> & cells = sel[std::size_t(t)];
            const std::int64_t want = width - (pos + 1 - tail_start);
            for (std::int64_t i = 0; i < n_blocks && std::int64_t(cells.size()) < want; ++i) {
                const std::int64_t b = score[std::size_t(i)].second;
                for (int j = 0; j < r && std::int64_t(cells.size()) < want; ++j) cells.push_back(std::int32_t(b * r + j));
            }
            for (std::int64_t p = tail_start; p <= pos; ++p) cells.push_back(std::int32_t(p));
            std::sort(cells.begin(), cells.end());
        });
        emit("indexer_q", il, T, std::int64_t(nih) * dim, qi.data());
        if (hook) {  // the selected positions of each query (as floats, -1 padded; all -1 = dense)
            std::vector<float> fs(std::size_t(T * width), -1.0f);
            for (std::int64_t t = 0; t < T; ++t) {
                const auto & c = sel[std::size_t(t)];
                for (std::size_t j = 0; j < c.size() && std::int64_t(j) < width; ++j) fs[std::size_t(t * width) + j] = float(c[j]);
            }
            emit("indexer_sel", il, T, width, fs.data());
        }
    }

    // ---------------------------------------------------------------------------- MoE + shared expert
    std::vector<float> ffn(const Layer & L, int il, const std::vector<float> & x, std::int64_t T) {
        const int E = cfg.n_embd, ne = cfg.n_expert, k = cfg.n_expert_used, F = cfg.n_ff_exp;
        std::vector<float> logits = matmul(L.router, x, T);  // [T][ne]
        live("ffn_moe_logits", il, T, ne, logits.data());

        // softmax gating, top-k by probability, weights renormalised over the selection
        std::vector<std::int32_t> sel(std::size_t(T * k));
        std::vector<float> wsel(std::size_t(T * k));
        for (std::int64_t t = 0; t < T; ++t) {
            const float * lg = logits.data() + t * ne;
            float mx = -INFINITY;
            for (int e = 0; e < ne; ++e) mx = std::max(mx, lg[e]);
            std::vector<float> p(static_cast<std::size_t>(ne));
            double sum = 0.0;
            for (int e = 0; e < ne; ++e) {
                p[std::size_t(e)] = std::exp(lg[e] - mx);
                sum += p[std::size_t(e)];
            }
            const float inv = float(1.0 / sum);
            for (auto & v : p) v *= inv;
            std::vector<std::int32_t> order(static_cast<std::size_t>(ne));
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](std::int32_t a, std::int32_t b) {
                return p[std::size_t(a)] != p[std::size_t(b)] ? p[std::size_t(a)] > p[std::size_t(b)] : a < b;
            });
            float wsum = 0.0f;
            for (int i = 0; i < k; ++i) wsum += p[std::size_t(order[std::size_t(i)])];
            wsum = std::max(wsum, 6.103515625e-5f);
            for (int i = 0; i < k; ++i) {
                sel[std::size_t(t * k + i)] = order[std::size_t(i)];
                float w = p[std::size_t(order[std::size_t(i)])] / wsum;
                if (cfg.expert_weights_scale != 0.0f && cfg.expert_weights_scale != 1.0f) w *= cfg.expert_weights_scale;
                wsel[std::size_t(t * k + i)] = w;
            }
        }
        if (hook) {
            std::vector<float> fsel(sel.size());
            for (std::size_t i = 0; i < sel.size(); ++i) fsel[i] = float(sel[i]);
            emit("ffn_moe_topk", il, T, k, fsel.data());
            emit("ffn_moe_weights", il, T, k, wsel.data());
        }

        // group the (token, slot) pairs by expert
        std::vector<std::vector<std::int32_t>> rows_of(static_cast<std::size_t>(ne));  // entries t*k + i
        for (std::int64_t i = 0; i < T * k; ++i) rows_of[std::size_t(sel[std::size_t(i)])].push_back(std::int32_t(i));
        std::vector<std::int32_t> active;
        std::vector<std::int64_t> base(std::size_t(ne), 0);  // first row of each expert in the packed buffers
        std::int64_t total = 0;
        for (int e = 0; e < ne; ++e) {
            if (rows_of[std::size_t(e)].empty()) continue;
            active.push_back(e);
            base[std::size_t(e)] = total;
            total += std::int64_t(rows_of[std::size_t(e)].size());
        }
        std::vector<float> xin(std::size_t(total * E)), gbuf(std::size_t(total * F)), ubuf(std::size_t(total * F)),
            dbuf(std::size_t(total * E));
        for (int e : active) {
            const auto & rs = rows_of[std::size_t(e)];
            for (std::size_t j = 0; j < rs.size(); ++j) {
                std::memcpy(xin.data() + (base[std::size_t(e)] + std::int64_t(j)) * E, x.data() + (rs[j] / k) * E,
                            std::size_t(E) * sizeof(float));
            }
        }
        if (opt.emulate != ReferenceOptions::Emulate::none) {
            const ActRound m = cpu_round(L.gate_exps->type);
            for (std::int64_t i = 0; i < total; ++i) round_activations(m, xin.data() + i * E, xin.data() + i * E, E);
        }
        // gate and up rows, in row chunks per expert
        const std::int64_t fc = (F + kRowBlock - 1) / kRowBlock, dc = (E + kRowBlock - 1) / kRowBlock;
        const std::int64_t n_act = std::int64_t(active.size());
        parallel_for(pool, n_act * fc * 2, [&](std::int64_t item, int tid) {
            const int e = active[std::size_t(item / (fc * 2))];
            const std::int64_t rem = item % (fc * 2), which = rem / fc, chunk = rem % fc;
            const Mat w = expert_of(which == 0 ? *L.gate_exps : *L.up_exps, e);
            const std::int64_t n0 = chunk * kRowBlock, n1 = std::min<std::int64_t>(F, n0 + kRowBlock);
            const std::int64_t b0 = base[std::size_t(e)], nt = std::int64_t(rows_of[std::size_t(e)].size());
            float * out = (which == 0 ? gbuf : ubuf).data() + b0 * F;
            gemm_rows(w, n0, n1, xin.data() + b0 * E, 0, nt, E, out, F, wbuf[std::size_t(tid)].data());
        });
        // swiglu: silu(gate) * up
        for (std::int64_t i = 0; i < total * F; ++i) gbuf[std::size_t(i)] = siluf_(gbuf[std::size_t(i)]) * ubuf[std::size_t(i)];
        if (opt.emulate != ReferenceOptions::Emulate::none) {
            const ActRound m = cpu_round(L.down_exps->type);
            for (std::int64_t i = 0; i < total; ++i) round_activations(m, gbuf.data() + i * F, gbuf.data() + i * F, F);
        }
        parallel_for(pool, n_act * dc, [&](std::int64_t item, int tid) {
            const int e = active[std::size_t(item / dc)];
            const std::int64_t chunk = item % dc;
            const Mat w = expert_of(*L.down_exps, e);
            const std::int64_t n0 = chunk * kRowBlock, n1 = std::min<std::int64_t>(E, n0 + kRowBlock);
            const std::int64_t b0 = base[std::size_t(e)], nt = std::int64_t(rows_of[std::size_t(e)].size());
            gemm_rows(w, n0, n1, gbuf.data() + b0 * F, 0, nt, F, dbuf.data() + b0 * E, E, wbuf[std::size_t(tid)].data());
        });
        // weighted sum in selection order: ((w0*d0 + w1*d1) + w2*d2) + ...
        std::vector<std::int64_t> where(std::size_t(T * k));
        for (int e : active) {
            const auto & rs = rows_of[std::size_t(e)];
            for (std::size_t j = 0; j < rs.size(); ++j) where[std::size_t(rs[j])] = base[std::size_t(e)] + std::int64_t(j);
        }
        std::vector<float> moe(std::size_t(T * E));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            float * m = moe.data() + t * E;
            for (int i = 0; i < k; ++i) {
                const float w = wsel[std::size_t(t * k + i)];
                const float * d = dbuf.data() + where[std::size_t(t * k + i)] * E;
                if (i == 0) {
                    for (int e = 0; e < E; ++e) m[e] = d[e] * w;
                } else {
                    for (int e = 0; e < E; ++e) m[e] = m[e] + d[e] * w;
                }
            }
        });
        live("ffn_moe_out", il, T, E, moe.data());

        // shared expert: down(silu(gate x) * up x) * sigmoid(gate_inp_shexp . x)
        std::vector<float> sg = matmul(L.sh_gate, x, T), su = matmul(L.sh_up, x, T);
        for (std::size_t i = 0; i < sg.size(); ++i) sg[i] = siluf_(sg[i]) * su[i];
        std::vector<float> sd = matmul(L.sh_down, sg, T);
        std::vector<float> out(std::size_t(T * E));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            const float gate = sigmoidf_(dot(L.sh_gate_inp.data(), x.data() + t * E, E));
            for (int e = 0; e < E; ++e) out[std::size_t(t * E + e)] = moe[std::size_t(t * E + e)] + sd[std::size_t(t * E + e)] * gate;
        });
        return out;
    }

    // ---------------------------------------------------------------------------- one chunk
    std::vector<float> forward_chunk(const std::int32_t * tokens, std::int64_t T, bool all_logits) {
        const int E = cfg.n_embd, hc = cfg.hc, HCD = hc * E;
        for (std::int64_t t = 0; t < T; ++t) {
            if (tokens[t] < 0 || tokens[t] >= cfg.n_vocab) throw std::runtime_error("reference: token id out of range");
            history.push_back(tokens[t]);
        }
        std::vector<float> x(std::size_t(T * E));
        parallel_for(pool, T, [&](std::int64_t t, int) {
            dequantize_row(tok_embd.type, tok_embd.data + std::size_t(tokens[t]) * tok_embd.rb, x.data() + t * E, E);
        });
        live("model.input_embed", -1, T, E, x.data());

        // the wide residual starts as hc copies of the embedding
        std::vector<float> res(std::size_t(T * HCD));
        for (std::int64_t t = 0; t < T; ++t) {
            for (int c = 0; c < hc; ++c) std::memcpy(res.data() + (t * hc + c) * E, x.data() + t * E, std::size_t(E) * sizeof(float));
        }

        std::vector<float> mixed, inject;
        for (int il = 0; il < cfg.n_layer; ++il) {
            const Layer & L = layers[std::size_t(il)];
            if (L.ple) ple(L, il, res, T);

            hc_mix(il, res, T, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, mixed, &inject);
            live("hc_attn_mixed", il, T, E, mixed.data());
            std::vector<float> out = L.recurrent ? deltanet(L, il, mixed, T) : attention(L, il, mixed, T);
            live(L.recurrent ? "linear_attn_out" : "attn_output", il, T, E, out.data());
            hc_combine(il, res, T, out, inject);

            hc_mix(il, res, T, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, mixed, &inject);
            live("hc_ffn_mixed", il, T, E, mixed.data());
            out = ffn(L, il, mixed, T);
            live("ffn_out", il, T, E, out.data());
            hc_combine(il, res, T, out, inject);
            live("l_out", il, T, HCD, res.data());
        }

        // the final mixer is the output norm
        const std::int64_t t_first = all_logits ? 0 : T - 1, n_out = T - t_first;
        std::vector<float> last(res.begin() + t_first * HCD, res.end());
        hc_mix(-1, last, n_out, hc_head_norm, hc_head_down, hc_head_up, nullptr, mixed, nullptr);
        if (hook) hook("result_norm", -1, n_past + t_first, n_out, E, mixed.data());
        std::vector<float> logits(std::size_t(n_out) * std::size_t(cfg.n_vocab));
        matmul(output, mixed.data(), n_out, E, logits.data(), cfg.n_vocab);
        if (hook) hook("result_output", -1, n_past + t_first, n_out, cfg.n_vocab, logits.data());
        n_past += T;
        return logits;
    }
};

// ================================================================================================

ReferenceModel::ReferenceModel(const GgufModel & model, ReferenceOptions options)
    : impl_(std::make_unique<Impl>(model, options)) {}

ReferenceModel::~ReferenceModel() = default;

std::vector<float> ReferenceModel::forward(const std::vector<std::int32_t> & tokens, bool all_logits) {
    if (tokens.empty()) throw std::runtime_error("reference: forward() needs at least one token");
    const std::int64_t n = std::int64_t(tokens.size());
    const std::int64_t chunk = std::max<std::int64_t>(1, impl_->opt.max_chunk);
    std::vector<float> all;
    std::vector<float> last;
    for (std::int64_t i = 0; i < n; i += chunk) {
        const std::int64_t T = std::min(chunk, n - i);
        const bool final_chunk = i + T == n;
        std::vector<float> lg = impl_->forward_chunk(tokens.data() + i, T, all_logits);
        if (all_logits) {
            all.insert(all.end(), lg.begin(), lg.end());
        } else if (final_chunk) {
            last = std::move(lg);
        }
    }
    return all_logits ? all : last;
}

void ReferenceModel::reset() { impl_->reset(); }

std::int64_t ReferenceModel::n_past() const { return impl_->n_past; }

const ReferenceConfig & ReferenceModel::config() const { return impl_->cfg; }

void ReferenceModel::set_activation_hook(ActivationHook hook) { impl_->hook = std::move(hook); }

void ReferenceModel::set_activation_override(ActivationOverride override_fn) { impl_->override_fn = std::move(override_fn); }

double ReferenceModel::self_test_matmul() {
    Impl & m = *impl_;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<Mat> mats = {m.layers[0].wqkv, m.layers[3].wq, m.layers[3].idx_q, m.layers[0].router, m.layers[0].hc_attn_up,
                             m.layers[0].hc_attn_inject, expert_of(*m.layers[0].gate_exps, 7), expert_of(*m.layers[0].down_exps, 300),
                             expert_of(*m.layers[2].up_exps, 5), m.output};
    double worst = 0.0;
    for (const Mat & full : mats) {
        Mat w = full;
        w.N = std::min<std::int64_t>(w.N, 203);  // a few row blocks plus a ragged tail
        for (std::int64_t T : {1, 3, 37}) {
            std::vector<float> X(std::size_t(T * w.K));
            for (auto & v : X) v = nd(rng);
            std::vector<float> Y(std::size_t(T * w.N));
            m.matmul(w, X.data(), T, w.K, Y.data(), w.N);
            std::vector<float> row(std::size_t(w.K));
            for (std::int64_t n = 0; n < w.N; ++n) {
                dequantize_row(w.type, w.data + std::size_t(n) * w.rb, row.data(), w.K);
                for (std::int64_t t = 0; t < T; ++t) {
                    double ref = 0.0, mag = 0.0;
                    for (std::int64_t k = 0; k < w.K; ++k) {
                        ref += double(row[std::size_t(k)]) * double(X[std::size_t(t * w.K + k)]);
                        mag += std::fabs(double(row[std::size_t(k)]) * double(X[std::size_t(t * w.K + k)]));
                    }
                    const double err = std::fabs(double(Y[std::size_t(t * w.N + n)]) - ref) / std::max(mag, 1e-30);
                    worst = std::max(worst, err);
                }
            }
        }
    }
    return worst;
}

}  // namespace ninfer::flashnext
