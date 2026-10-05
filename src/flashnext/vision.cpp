#include "flashnext/vision.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/vision.h"
#include "flashnext/gguf.h"
#include "flashnext/quants.h"

namespace ninfer::flashnext {

namespace fc = cuda;
using fc::check;

namespace {

void cublas_check(cublasStatus_t st, const char * what) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("vision: ") + what + ": cuBLAS status " + std::to_string(int(st)));
}

constexpr std::size_t kAlign = 256;
std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// A blob of sections (F16 matrices, F32 vectors) laid out back to back, 256-byte aligned; the same layout on the host
// (pinned) and in a device staging buffer.
struct Blob {
    std::size_t bytes = 0;
    std::size_t add(std::size_t n) {
        const std::size_t off = bytes;
        bytes = align_up(bytes + n);
        return off;
    }
};

// per layer: the four matrices (F16) and the vectors (F32)
struct LayerLayout {
    std::size_t qkv_w, out_w, up_w, down_w, qkv_b, out_b, up_b, down_b, ln1_w, ln1_b, ln2_w, ln2_b;
};
struct MergerLayout {
    std::size_t post_w, post_b, mm0_w, mm0_b, mm2_w, mm2_b;
};
struct GlobalLayout {
    std::size_t patch_w, patch_b, pos;
};

}  // namespace

struct VisionEncoder::Impl {
    VisionConfig cfg;
    bool exact_merger_gelu = true;
    int side = 0;  // position grid per side
    std::size_t layer_bytes = 0, merger_bytes = 0, global_bytes = 0, stage_bytes = 0;
    LayerLayout ll{};
    MergerLayout ml{};
    GlobalLayout gl{};
    std::uint8_t * host = nullptr;  // pinned: [global][layer 0] .. [layer L-1][merger]
    std::size_t host_bytes = 0;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr, copy = nullptr;
    cudaEvent_t ev_copied[2] = {}, ev_done[2] = {}, ev_global = nullptr, ev_t0 = nullptr, ev_t1 = nullptr;

    static constexpr std::size_t kBlasWorkspace = std::size_t(32) << 20;

    std::size_t blob_offset(int b) const {  // b in 0..layers: a layer, then the merger
        return global_bytes + std::size_t(b) * layer_bytes;
    }
    std::size_t max_matrix() const {
        const std::size_t h = std::size_t(cfg.hidden), m = std::size_t(cfg.merge * cfg.merge) * h;
        return std::max({3 * h * h, h * std::size_t(cfg.ff), m * m, std::size_t(cfg.out_dim) * m, h * std::size_t(fc::kVisPatchIn)});
    }

    explicit Impl(const std::string & path) {
        GgufModel g({path});
        auto req = [&](bool ok, const std::string & what) {
            if (!ok) throw std::runtime_error("vision: " + path + ": " + what);
        };
        req(g.has("general.architecture") && g.get_string("general.architecture") == "clip", "not a clip (mmproj) GGUF");
        req(g.has("clip.projector_type") && g.get_string("clip.projector_type") == "qwen3vl_merger", "projector is not qwen3vl_merger");
        cfg.image_size = int(g.get_int("clip.vision.image_size"));
        cfg.patch = int(g.get_int("clip.vision.patch_size"));
        cfg.hidden = int(g.get_int("clip.vision.embedding_length"));
        cfg.ff = int(g.get_int("clip.vision.feed_forward_length"));
        cfg.layers = int(g.get_int("clip.vision.block_count"));
        cfg.heads = int(g.get_int("clip.vision.attention.head_count"));
        cfg.merge = int(g.get_int("clip.vision.spatial_merge_size"));
        cfg.out_dim = int(g.get_int("clip.vision.projection_dim"));
        cfg.eps = float(g.get_float("clip.vision.attention.layer_norm_epsilon"));
        cfg.mean_std = "0.5,0.5,0.5/0.5,0.5,0.5";
        req(cfg.hidden == fc::kVisHidden && cfg.heads == fc::kVisHeads && cfg.ff == fc::kVisFF && cfg.patch == 16 && cfg.merge == 2,
            "vision tower shapes differ from Qwen3-VL's (1152 wide, 16 heads, MLP 4304, patch 16, merge 2)");
        req(cfg.image_size % cfg.patch == 0 && cfg.layers > 0 && cfg.out_dim > 0, "invalid image size, depth or projection");
        side = cfg.image_size / cfg.patch;
        if (g.has("clip.vision.is_deepstack_layers"))
            for (std::int64_t v : g.get_ints("clip.vision.is_deepstack_layers")) req(v == 0, "deepstack layers are not supported");

        const std::size_t h = std::size_t(cfg.hidden), f = std::size_t(cfg.ff), m = 4 * h;
        Blob lb;
        ll.qkv_w = lb.add(3 * h * h * 2);
        ll.out_w = lb.add(h * h * 2);
        ll.up_w = lb.add(f * h * 2);
        ll.down_w = lb.add(h * f * 2);
        ll.qkv_b = lb.add(3 * h * 4);
        ll.out_b = lb.add(h * 4);
        ll.up_b = lb.add(f * 4);
        ll.down_b = lb.add(h * 4);
        ll.ln1_w = lb.add(h * 4);
        ll.ln1_b = lb.add(h * 4);
        ll.ln2_w = lb.add(h * 4);
        ll.ln2_b = lb.add(h * 4);
        layer_bytes = lb.bytes;
        Blob mb;
        ml.post_w = mb.add(h * 4);
        ml.post_b = mb.add(h * 4);
        ml.mm0_w = mb.add(m * m * 2);
        ml.mm0_b = mb.add(m * 4);
        ml.mm2_w = mb.add(std::size_t(cfg.out_dim) * m * 2);
        ml.mm2_b = mb.add(std::size_t(cfg.out_dim) * 4);
        merger_bytes = mb.bytes;
        Blob gb;
        gl.patch_w = gb.add(h * fc::kVisPatchIn * 2);
        gl.patch_b = gb.add(h * 4);
        gl.pos = gb.add(std::size_t(side) * side * h * 4);
        global_bytes = gb.bytes;
        stage_bytes = std::max(layer_bytes, merger_bytes);
        host_bytes = global_bytes + std::size_t(cfg.layers) * layer_bytes + merger_bytes;
        check(cudaMallocHost(reinterpret_cast<void **>(&host), host_bytes), "vision: pinned weights");

        // tensors into the blobs: matrices must be F16 ([N][K] rows), vectors F32 or F16
        auto matrix = [&](const std::string & name, std::int64_t k, std::int64_t n, std::uint8_t * dst) {
            const GgufTensor & t = g.tensor(name);
            req(t.type == GgufType::F16, name + " is not F16 (use the F16 mmproj)");
            req(t.shape.size() == 2 && t.shape[0] == k && t.shape[1] == n, name + " has an unexpected shape");
            std::memcpy(dst, t.data, std::size_t(k * n) * 2);
        };
        auto vector = [&](const std::string & name, std::int64_t n, std::uint8_t * dst) {
            const GgufTensor & t = g.tensor(name);
            req(t.elements() == n && (t.type == GgufType::F32 || t.type == GgufType::F16), name + " has an unexpected shape or type");
            dequantize_row(t.type, t.data, reinterpret_cast<float *>(dst), n);
        };
        std::uint8_t * G = host;
        {
            // the two temporal halves of the patch convolution side by side: column c*512 + t*256 + y*16 + x of row o
            // is weight t's [o][c][y][x] (ggml shape [16, 16, 3, 1152])
            const GgufTensor & w0 = g.tensor("v.patch_embd.weight");
            const GgufTensor & w1 = g.tensor("v.patch_embd.weight.1");
            const std::vector<std::int64_t> want{16, 16, 3, std::int64_t(h)};
            req(w0.type == GgufType::F16 && w1.type == GgufType::F16 && w0.shape == want && w1.shape == want, "unexpected patch embedding");
            auto * dst = reinterpret_cast<std::uint16_t *>(G + gl.patch_w);
            for (std::size_t o = 0; o < h; ++o)
                for (int c = 0; c < 3; ++c)
                    for (int t = 0; t < 2; ++t) {
                        const auto * src = reinterpret_cast<const std::uint16_t *>((t ? w1 : w0).data) + (o * 3 + std::size_t(c)) * 256;
                        std::memcpy(dst + o * fc::kVisPatchIn + std::size_t(c) * 512 + std::size_t(t) * 256, src, 256 * 2);
                    }
            vector("v.patch_embd.bias", std::int64_t(h), G + gl.patch_b);
            const GgufTensor & pos = g.tensor("v.position_embd.weight");
            req(pos.shape.size() == 2 && pos.shape[0] == std::int64_t(h) && pos.shape[1] == std::int64_t(side) * side, "unexpected position embedding");
            vector("v.position_embd.weight", std::int64_t(h) * side * side, G + gl.pos);
        }
        for (int il = 0; il < cfg.layers; ++il) {
            std::uint8_t * L = host + blob_offset(il);
            const std::string b = "v.blk." + std::to_string(il) + ".";
            matrix(b + "attn_qkv.weight", std::int64_t(h), std::int64_t(3 * h), L + ll.qkv_w);
            matrix(b + "attn_out.weight", std::int64_t(h), std::int64_t(h), L + ll.out_w);
            matrix(b + "ffn_up.weight", std::int64_t(h), std::int64_t(f), L + ll.up_w);
            matrix(b + "ffn_down.weight", std::int64_t(f), std::int64_t(h), L + ll.down_w);
            vector(b + "attn_qkv.bias", std::int64_t(3 * h), L + ll.qkv_b);
            vector(b + "attn_out.bias", std::int64_t(h), L + ll.out_b);
            vector(b + "ffn_up.bias", std::int64_t(f), L + ll.up_b);
            vector(b + "ffn_down.bias", std::int64_t(h), L + ll.down_b);
            vector(b + "ln1.weight", std::int64_t(h), L + ll.ln1_w);
            vector(b + "ln1.bias", std::int64_t(h), L + ll.ln1_b);
            vector(b + "ln2.weight", std::int64_t(h), L + ll.ln2_w);
            vector(b + "ln2.bias", std::int64_t(h), L + ll.ln2_b);
        }
        req(!g.find("v.blk." + std::to_string(cfg.layers) + ".ln1.weight"), "more layers than block_count");
        {
            std::uint8_t * M = host + blob_offset(cfg.layers);
            vector("v.post_ln.weight", std::int64_t(h), M + ml.post_w);
            vector("v.post_ln.bias", std::int64_t(h), M + ml.post_b);
            matrix("mm.0.weight", std::int64_t(m), std::int64_t(m), M + ml.mm0_w);
            vector("mm.0.bias", std::int64_t(m), M + ml.mm0_b);
            matrix("mm.2.weight", std::int64_t(m), std::int64_t(cfg.out_dim), M + ml.mm2_w);
            vector("mm.2.bias", std::int64_t(cfg.out_dim), M + ml.mm2_b);
        }
        req(!g.find("v.pre_ln.weight") && !g.find("mm.input_norm.weight"), "unexpected pre-norm tensors (not Qwen3-VL's layout)");

        fc::vision_init();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "vision: stream");
        check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking), "vision: stream");
        for (int i = 0; i < 2; ++i) {
            check(cudaEventCreateWithFlags(&ev_copied[i], cudaEventDisableTiming), "vision: event");
            check(cudaEventCreateWithFlags(&ev_done[i], cudaEventDisableTiming), "vision: event");
        }
        check(cudaEventCreateWithFlags(&ev_global, cudaEventDisableTiming), "vision: event");
        check(cudaEventCreate(&ev_t0), "vision: event");
        check(cudaEventCreate(&ev_t1), "vision: event");
        cublas_check(cublasCreate(&blas), "cublasCreate");
        cublas_check(cublasSetStream(blas, stream), "cublasSetStream");
        cublas_check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");  // FP32, no TF32
    }

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        if (copy) cudaStreamSynchronize(copy);
        if (blas) cublasDestroy(blas);
        for (cudaEvent_t e : {ev_copied[0], ev_copied[1], ev_done[0], ev_done[1], ev_global, ev_t0, ev_t1})
            if (e) cudaEventDestroy(e);
        if (copy) cudaStreamDestroy(copy);
        if (stream) cudaStreamDestroy(stream);
        if (host) cudaFreeHost(host);
    }

    // device workspace: [global][stage 0][stage 1][FP32 weight][cuBLAS workspace][x][A][B]
    struct Work {
        std::uint8_t *global, *stage[2];
        float *wf, *x, *a, *b;
        void * blas_ws;
        std::size_t bytes;
    };
    Work carve(void * base, int P) const {
        Work w{};
        std::uint8_t * p = static_cast<std::uint8_t *>(base);
        std::size_t off = 0;
        auto take = [&](std::size_t n) {
            std::uint8_t * r = p ? p + off : nullptr;
            off = align_up(off + n);
            return r;
        };
        const std::size_t h = std::size_t(cfg.hidden), n = std::size_t(P);
        w.global = take(global_bytes);
        w.stage[0] = take(stage_bytes);
        w.stage[1] = take(stage_bytes);
        w.wf = reinterpret_cast<float *>(take(max_matrix() * 4));
        w.blas_ws = take(kBlasWorkspace);
        w.x = reinterpret_cast<float *>(take(n * h * 4));
        w.a = reinterpret_cast<float *>(take(n * std::max<std::size_t>(std::size_t(cfg.ff), fc::kVisPatchIn) * 4));
        w.b = reinterpret_cast<float *>(take(n * h * 4));
        w.bytes = off;
        return w;
    }

    // y [rows][n] = x [rows][k] w^T (w F16 [n][k] in device memory) + beta * y
    void linear(const std::uint8_t * w16, int n, int k, const float * x, float * y, int rows, float * wf, float beta) {
        fc::f16_to_f32(reinterpret_cast<const half *>(w16), wf, std::size_t(n) * k, stream);
        const float one = 1.0f;
        cublas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, wf, k, x, k, &beta, y, n), "cublasSgemm");
    }

    VisionTiming encode(const float * patches, int gh, int gw, void * workspace, float * out) {
        const auto w0 = std::chrono::steady_clock::now();
        if (gh < 2 || gw < 2 || gh % 2 || gw % 2) throw std::runtime_error("vision: the patch grid must be even in both directions");
        const int P = gh * gw, T = P / 4, H = cfg.hidden, L = cfg.layers;
        const Work w = carve(workspace, P);
        cublas_check(cublasSetWorkspace(blas, w.blas_ws, kBlasWorkspace), "cublasSetWorkspace");
        // weights: the global blob and the first two blobs now, then blob b + 2 into the stage blob b used
        auto blob = [&](int b) { return host + blob_offset(b); };
        auto blob_bytes = [&](int b) { return b < L ? layer_bytes : merger_bytes; };
        check(cudaEventRecord(ev_t0, stream), "vision");
        check(cudaStreamWaitEvent(copy, ev_t0, 0), "vision");  // the workspace is free (the caller's stream is idle)
        check(cudaMemcpyAsync(w.global, host, global_bytes, cudaMemcpyHostToDevice, copy), "vision: weights");
        check(cudaEventRecord(ev_global, copy), "vision");
        for (int b = 0; b < 2 && b <= L; ++b) {
            check(cudaMemcpyAsync(w.stage[b], blob(b), blob_bytes(b), cudaMemcpyHostToDevice, copy), "vision: weights");
            check(cudaEventRecord(ev_copied[b], copy), "vision");
        }
        // patch embedding (+ bias, + the position embedding resized to the grid)
        check(cudaMemcpyAsync(w.a, patches, std::size_t(P) * fc::kVisPatchIn * sizeof(float), cudaMemcpyHostToDevice, stream), "vision: patches");
        check(cudaStreamWaitEvent(stream, ev_global, 0), "vision");
        linear(w.global + gl.patch_w, H, fc::kVisPatchIn, w.a, w.x, P, w.wf, 0.0f);
        fc::vis_embed_add(w.x, reinterpret_cast<const float *>(w.global + gl.patch_b), reinterpret_cast<const float *>(w.global + gl.pos), side,
                          gh, gw, stream);
        for (int il = 0; il < L; ++il) {
            const int s = il & 1;
            check(cudaStreamWaitEvent(stream, ev_copied[s], 0), "vision");
            const std::uint8_t * W = w.stage[s];
            auto vec = [&](std::size_t off) { return reinterpret_cast<const float *>(W + off); };
            fc::vis_layer_norm(w.x, vec(ll.ln1_w), vec(ll.ln1_b), w.b, P, H, cfg.eps, stream);
            linear(W + ll.qkv_w, 3 * H, H, w.b, w.a, P, w.wf, 0.0f);
            fc::vis_qkv_rope(w.a, vec(ll.qkv_b), gw, P, stream);
            fc::vis_attention(w.a, w.b, P, stream);
            linear(W + ll.out_w, H, H, w.b, w.x, P, w.wf, 1.0f);  // the residual adds in the product
            fc::vis_bias_add(w.x, vec(ll.out_b), P, H, stream);
            fc::vis_layer_norm(w.x, vec(ll.ln2_w), vec(ll.ln2_b), w.b, P, H, cfg.eps, stream);
            linear(W + ll.up_w, cfg.ff, H, w.b, w.a, P, w.wf, 0.0f);
            fc::vis_bias_gelu(w.a, vec(ll.up_b), P, cfg.ff, false, stream);
            linear(W + ll.down_w, H, cfg.ff, w.a, w.x, P, w.wf, 1.0f);
            fc::vis_bias_add(w.x, vec(ll.down_b), P, H, stream);
            check(cudaEventRecord(ev_done[s], stream), "vision");
            if (il + 2 <= L) {  // the stage this layer used takes blob il + 2 (a layer or the merger)
                check(cudaStreamWaitEvent(copy, ev_done[s], 0), "vision");
                check(cudaMemcpyAsync(w.stage[s], blob(il + 2), blob_bytes(il + 2), cudaMemcpyHostToDevice, copy), "vision: weights");
                check(cudaEventRecord(ev_copied[s], copy), "vision");
            }
        }
        {
            // merger: LayerNorm per patch, then each 2x2 block's 4 patches as one 4608-wide row through the MLP
            const int s = L & 1;
            check(cudaStreamWaitEvent(stream, ev_copied[s], 0), "vision");
            const std::uint8_t * W = w.stage[s];
            auto vec = [&](std::size_t off) { return reinterpret_cast<const float *>(W + off); };
            const int MW = 4 * H;
            fc::vis_layer_norm(w.x, vec(ml.post_w), vec(ml.post_b), w.b, P, H, cfg.eps, stream);
            linear(W + ml.mm0_w, MW, MW, w.b, w.a, T, w.wf, 0.0f);
            fc::vis_bias_gelu(w.a, vec(ml.mm0_b), T, MW, exact_merger_gelu, stream);
            linear(W + ml.mm2_w, cfg.out_dim, MW, w.a, w.x, T, w.wf, 0.0f);
            fc::vis_bias_add(w.x, vec(ml.mm2_b), T, cfg.out_dim, stream);
        }
        check(cudaEventRecord(ev_t1, stream), "vision");
        check(cudaMemcpyAsync(out, w.x, std::size_t(T) * cfg.out_dim * sizeof(float), cudaMemcpyDeviceToHost, stream), "vision: output");
        check(cudaStreamSynchronize(stream), "vision");
        check(cudaStreamSynchronize(copy), "vision");
        VisionTiming t;
        float ms = 0;
        check(cudaEventElapsedTime(&ms, ev_t0, ev_t1), "vision");
        t.gpu_ms = ms;
        t.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        return t;
    }
};

VisionEncoder::VisionEncoder(const std::string & mmproj_path) : impl_(std::make_unique<Impl>(mmproj_path)) {}
VisionEncoder::~VisionEncoder() = default;
const VisionConfig & VisionEncoder::config() const { return impl_->cfg; }
int VisionEncoder::patch_values() const { return fc::kVisPatchIn; }
std::size_t VisionEncoder::workspace_bytes(int patches) const { return impl_->carve(nullptr, patches).bytes; }
VisionTiming VisionEncoder::encode(const float * patches, int grid_h, int grid_w, void * workspace, float * out) {
    return impl_->encode(patches, grid_h, grid_w, workspace, out);
}
void VisionEncoder::set_exact_merger_gelu(bool exact) { impl_->exact_merger_gelu = exact; }

}  // namespace ninfer::flashnext
