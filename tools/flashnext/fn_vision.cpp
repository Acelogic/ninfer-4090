// Checks and timings of the Flash-Next vision path: the vision encoder (against llama.cpp's embeddings and a
// double-precision CPU reference) and image prompts through the engine (against llama.cpp's logits and the FP32
// reference model).
//
// Encoder:  fn_vision --mmproj mmproj-F16.gguf --image img.ppm [--embd-out ours.sve] [--compare llama.sve] [--cpu-ref]
//                     [--repeat N] [--merger-gelu erf|tanh]
//   The image is a binary PPM (P6, 8-bit) whose sides are multiples of 32 (no resize, so both implementations see the
//   same pixels); patches are normalized as llama.cpp does, (u / 255 - 0.5) / 0.5. .sve files: int32 {'SVE1', n, nx,
//   ny, n_embd} then float32 [n][n_embd] (the layout of Strata's strata-vision, which tools/flashnext's llama.cpp
//   companion vision_ref also writes).
// Prompt:   fn_vision --mmproj ... -m model-00001-of-00003.gguf --case case.txt [--llama-embd] [-n 16] [--ctx N]
//                     [--mtp head.gguf [--draft K]] [--pieces N] [--compare-ref] [--test-snapshot] [--cache-mib N]
//                     [--routing-stats file]
//   case.txt (written by vision_ref): "tokens <ids>", "pos_t/pos_h/pos_w <n values>", one "image <first token> <nx> <ny>
//   <image.ppm> <llama.sve>" per image, "greedy <ids>" and "logits <file of n_vocab float32>" (llama.cpp's). The
//   images are encoded with our encoder (--llama-embd: llama.cpp's rows instead), the prompt runs with their rows and
//   positions, and the first logits and the greedy continuation are compared with llama.cpp's (and the FP32
//   reference's with --compare-ref).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "flashnext/engine.h"
#include "flashnext/gguf.h"
#include "flashnext/quants.h"
#include "flashnext/reference.h"
#include "flashnext/vision.h"

using namespace ninfer::flashnext;
namespace fs = std::filesystem;

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

std::vector<std::string> shard_paths(const std::string & first) {
    std::smatch mm;
    const std::string name = fs::path(first).filename().string();
    static const std::regex pat(R"((.*)-(\d{5})-of-(\d{5})\.gguf)");
    if (!std::regex_match(name, mm, pat)) return {first};
    const int count = std::stoi(mm[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= count; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, count);
        out.push_back((fs::path(first).parent_path() / (mm[1].str() + buf)).string());
    }
    return out;
}

struct Image {
    int w = 0, h = 0;
    std::vector<std::uint8_t> rgb;
};

Image read_ppm(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string magic;
    int maxv = 0;
    Image im;
    f >> magic >> im.w >> im.h >> maxv;
    f.get();
    if (magic != "P6" || maxv != 255 || im.w <= 0 || im.h <= 0) throw std::runtime_error(path + " is not an 8-bit binary PPM");
    im.rgb.resize(std::size_t(im.w) * im.h * 3);
    f.read(reinterpret_cast<char *>(im.rgb.data()), std::streamsize(im.rgb.size()));
    if (!f) throw std::runtime_error(path + " is truncated");
    return im;
}

// Patches in the frontend's merge order, each [channel][frame][y][x] with the frame repeated, normalized as llama.cpp.
std::vector<float> patches_of(const Image & im, int & gh, int & gw) {
    if (im.w % 32 || im.h % 32) throw std::runtime_error("the image sides must be multiples of 32 (resize it first)");
    gh = im.h / 16;
    gw = im.w / 16;
    std::vector<float> out;
    out.reserve(std::size_t(gh) * gw * 1536);
    for (int by = 0; by < gh / 2; ++by)
        for (int bx = 0; bx < gw / 2; ++bx)
            for (int my = 0; my < 2; ++my)
                for (int mx = 0; mx < 2; ++mx) {
                    const int py = by * 2 + my, px = bx * 2 + mx;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < 2; ++t)
                            for (int y = 0; y < 16; ++y)
                                for (int x = 0; x < 16; ++x) {
                                    const std::uint8_t u = im.rgb[(std::size_t(py * 16 + y) * im.w + std::size_t(px * 16 + x)) * 3 + c];
                                    out.push_back((float(u) / 255.0f - 0.5f) / 0.5f);
                                }
                }
    return out;
}

struct Sve {
    int n = 0, nx = 0, ny = 0, dim = 0;
    std::vector<float> rows;
};

Sve read_sve(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::int32_t hdr[5] = {};
    if (!f.read(reinterpret_cast<char *>(hdr), sizeof(hdr)) || hdr[0] != 0x31455653) throw std::runtime_error(path + " is not an SVE1 file");
    Sve s{hdr[1], hdr[2], hdr[3], hdr[4], {}};
    s.rows.resize(std::size_t(s.n) * s.dim);
    if (!f.read(reinterpret_cast<char *>(s.rows.data()), std::streamsize(s.rows.size() * sizeof(float)))) throw std::runtime_error(path + " is truncated");
    return s;
}

void write_sve(const std::string & path, const Sve & s) {
    std::ofstream f(path, std::ios::binary);
    const std::int32_t hdr[5] = {0x31455653, s.n, s.nx, s.ny, s.dim};
    f.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char *>(s.rows.data()), std::streamsize(s.rows.size() * sizeof(float)));
}

// relative L2 error of a against b over all values, the worst row's, the largest absolute difference, the mean cosine
void compare(const char * what, const std::vector<float> & a, const std::vector<float> & b, int dim) {
    if (a.size() != b.size()) {
        std::printf("%s: sizes differ (%zu vs %zu)\n", what, a.size(), b.size());
        return;
    }
    double num = 0, den = 0, worst_row = 0, maxabs = 0, cos_sum = 0;
    const std::size_t rows = a.size() / std::size_t(dim);
    for (std::size_t r = 0; r < rows; ++r) {
        double n = 0, d = 0, ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < dim; ++i) {
            const double x = a[r * dim + i], y = b[r * dim + i];
            n += (x - y) * (x - y);
            d += y * y;
            ab += x * y;
            aa += x * x;
            bb += y * y;
            maxabs = std::max(maxabs, std::fabs(x - y));
        }
        num += n;
        den += d;
        worst_row = std::max(worst_row, std::sqrt(n / std::max(d, 1e-30)));
        cos_sum += ab / std::sqrt(std::max(aa * bb, 1e-60));
    }
    std::printf("%s: rel L2 %.3e, worst row %.3e, max |diff| %.3e, mean cosine %.9f (%zu rows)\n", what, std::sqrt(num / std::max(den, 1e-30)),
                worst_row, maxabs, cos_sum / double(rows), rows);
}

// ------------------------------------------------------------------------------------------------
// Double-precision CPU reference of the encoder (same formulas as src/flashnext/cuda/vision.cu).

struct CpuVision {
    const GgufModel & g;
    int H = 1152, F = 4304, NH = 16, D = 72, L = 0, side = 0, out_dim = 0;
    double eps = 1e-6;
    bool exact_merger_gelu = true;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());

    explicit CpuVision(const GgufModel & m) : g(m) {
        L = int(g.get_int("clip.vision.block_count"));
        side = int(g.get_int("clip.vision.image_size") / g.get_int("clip.vision.patch_size"));
        out_dim = int(g.get_int("clip.vision.projection_dim"));
        eps = g.get_float("clip.vision.attention.layer_norm_epsilon");
    }
    std::vector<float> vec(const std::string & name) const {
        const GgufTensor & t = g.tensor(name);
        std::vector<float> v(std::size_t(t.elements()));
        dequantize_row(t.type, t.data, v.data(), t.elements());
        return v;
    }
    template <class Fn>
    void par(std::size_t n, Fn fn) const {
        std::vector<std::thread> pool;
        for (unsigned k = 0; k < threads; ++k)
            pool.emplace_back([&, k] {
                for (std::size_t i = k; i < n; i += threads) fn(i);
            });
        for (auto & t : pool) t.join();
    }
    // y [rows][n] = x [rows][k] W^T + b (W [n][k])
    std::vector<double> linear(const std::vector<float> & W, const std::vector<float> & b, const std::vector<double> & x, std::size_t rows,
                               int n, int k) const {
        std::vector<double> y(rows * std::size_t(n));
        par(rows, [&](std::size_t r) {
            for (int o = 0; o < n; ++o) {
                double s = 0;
                const float * w = W.data() + std::size_t(o) * k;
                const double * xr = x.data() + r * std::size_t(k);
                for (int i = 0; i < k; ++i) s += double(w[i]) * xr[i];
                y[r * n + o] = s + (b.empty() ? 0.0 : double(b[std::size_t(o)]));
            }
        });
        return y;
    }
    void layer_norm(std::vector<double> & x, std::size_t rows, int n, const std::vector<float> & w, const std::vector<float> & b) const {
        for (std::size_t r = 0; r < rows; ++r) {
            double * v = x.data() + r * n;
            double m = 0, var = 0;
            for (int i = 0; i < n; ++i) m += v[i];
            m /= n;
            for (int i = 0; i < n; ++i) var += (v[i] - m) * (v[i] - m);
            var /= n;
            const double inv = 1.0 / std::sqrt(var + eps);
            for (int i = 0; i < n; ++i) v[i] = (v[i] - m) * inv * w[std::size_t(i)] + b[std::size_t(i)];
        }
    }
    static void cell(int p, int gw, int & row, int & col) {
        const int block = p / 4, w = p % 4, bw = gw / 2;
        row = 2 * (block / bw) + w / 2;
        col = 2 * (block % bw) + w % 2;
    }

    std::vector<float> encode(const std::vector<float> & patches, int gh, int gw) const {
        const std::size_t P = std::size_t(gh) * gw;
        // patch embedding: both frames' weights against the [c][t][y][x] patch
        std::vector<float> Wp(std::size_t(H) * 1536);
        {
            const std::vector<float> w0 = vec("v.patch_embd.weight"), w1 = vec("v.patch_embd.weight.1");
            for (int o = 0; o < H; ++o)
                for (int c = 0; c < 3; ++c)
                    for (int t = 0; t < 2; ++t)
                        for (int i = 0; i < 256; ++i)
                            Wp[std::size_t(o) * 1536 + std::size_t(c) * 512 + std::size_t(t) * 256 + std::size_t(i)] =
                                (t ? w1 : w0)[(std::size_t(o) * 3 + std::size_t(c)) * 256 + std::size_t(i)];
        }
        std::vector<double> in(patches.begin(), patches.end());
        std::vector<double> x = linear(Wp, vec("v.patch_embd.bias"), in, P, H, 1536);
        {
            const std::vector<float> pos = vec("v.position_embd.weight");
            const float sf1 = gh > 1 ? float(gh - 1) / float(side - 1) : float(gh) / float(side);
            const float sf0 = gw > 1 ? float(gw - 1) / float(side - 1) : float(gw) / float(side);
            for (std::size_t p = 0; p < P; ++p) {
                int row = 0, col = 0;
                cell(int(p), gw, row, col);
                const float y = float(row) / sf1, xf = float(col) / sf0;
                const int y0 = std::min(std::max(int(std::floor(y)), 0), side - 1), y1 = std::min(y0 + 1, side - 1);
                const int x0 = std::min(std::max(int(std::floor(xf)), 0), side - 1), x1 = std::min(x0 + 1, side - 1);
                const double dy = std::clamp(double(y - float(y0)), 0.0, 1.0), dx = std::clamp(double(xf - float(x0)), 0.0, 1.0);
                for (int i = 0; i < H; ++i) {
                    auto at = [&](int yy, int xx) { return double(pos[(std::size_t(yy) * side + std::size_t(xx)) * H + std::size_t(i)]); };
                    x[p * H + i] += at(y0, x0) * (1 - dx) * (1 - dy) + at(y0, x1) * dx * (1 - dy) + at(y1, x0) * (1 - dx) * dy + at(y1, x1) * dx * dy;
                }
            }
        }
        double freq[18];
        for (int j = 0; j < 18; ++j) freq[j] = std::pow(10000.0, -double(j) / 18.0);
        for (int il = 0; il < L; ++il) {
            const std::string b = "v.blk." + std::to_string(il) + ".";
            std::vector<double> h = x;
            layer_norm(h, P, H, vec(b + "ln1.weight"), vec(b + "ln1.bias"));
            std::vector<double> qkv = linear(vec(b + "attn_qkv.weight"), vec(b + "attn_qkv.bias"), h, P, 3 * H, H);
            for (std::size_t p = 0; p < P; ++p) {
                int row = 0, col = 0;
                cell(int(p), gw, row, col);
                for (int part = 0; part < 2; ++part)
                    for (int hd = 0; hd < NH; ++hd)
                        for (int i = 0; i < 36; ++i) {
                            const double th = double(i < 18 ? row : col) * freq[i % 18];
                            double * v = qkv.data() + p * 3 * H + std::size_t(part) * H + std::size_t(hd) * D;
                            const double a = v[i], c = v[i + 36];
                            v[i] = a * std::cos(th) - c * std::sin(th);
                            v[i + 36] = a * std::sin(th) + c * std::cos(th);
                        }
            }
            std::vector<double> att(P * H);
            const double scale = 1.0 / std::sqrt(double(D));
            par(P * std::size_t(NH), [&](std::size_t item) {
                const std::size_t p = item / NH, hd = item % NH;
                const double * q = qkv.data() + p * 3 * H + hd * D;
                std::vector<double> s(P);
                double mx = -1e300;
                for (std::size_t j = 0; j < P; ++j) {
                    const double * k = qkv.data() + j * 3 * H + H + hd * D;
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += q[i] * k[i];
                    s[j] = d * scale;
                    mx = std::max(mx, s[j]);
                }
                double sum = 0;
                for (auto & v : s) sum += (v = std::exp(v - mx));
                double * o = att.data() + p * H + hd * D;
                for (int i = 0; i < D; ++i) o[i] = 0;
                for (std::size_t j = 0; j < P; ++j) {
                    const double * v = qkv.data() + j * 3 * H + 2 * H + hd * D;
                    for (int i = 0; i < D; ++i) o[i] += s[j] / sum * v[i];
                }
            });
            const std::vector<double> o = linear(vec(b + "attn_out.weight"), vec(b + "attn_out.bias"), att, P, H, H);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
            h = x;
            layer_norm(h, P, H, vec(b + "ln2.weight"), vec(b + "ln2.bias"));
            std::vector<double> up = linear(vec(b + "ffn_up.weight"), vec(b + "ffn_up.bias"), h, P, F, H);
            for (auto & v : up) v = 0.5 * v * (1.0 + std::tanh(0.79788456080286535588 * (v + 0.044715 * v * v * v)));
            const std::vector<double> dn = linear(vec(b + "ffn_down.weight"), vec(b + "ffn_down.bias"), up, P, H, F);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] += dn[i];
            std::fprintf(stderr, "\rcpu reference: layer %d/%d", il + 1, L);
        }
        std::fprintf(stderr, "\n");
        layer_norm(x, P, H, vec("v.post_ln.weight"), vec("v.post_ln.bias"));
        std::vector<double> m = linear(vec("mm.0.weight"), vec("mm.0.bias"), x, P / 4, 4 * H, 4 * H);
        for (auto & v : m)
            v = exact_merger_gelu ? 0.5 * v * (1.0 + std::erf(v * 0.70710678118654752440))
                                  : 0.5 * v * (1.0 + std::tanh(0.79788456080286535588 * (v + 0.044715 * v * v * v)));
        const std::vector<double> y = linear(vec("mm.2.weight"), vec("mm.2.bias"), m, P / 4, out_dim, 4 * H);
        return std::vector<float>(y.begin(), y.end());
    }
};

std::vector<std::int32_t> parse_ints(const std::string & s) {
    std::vector<std::int32_t> v;
    std::istringstream in(s);
    long long x = 0;
    while (in >> x) v.push_back(std::int32_t(x));
    return v;
}

struct Case {
    std::vector<std::int32_t> tokens, pos[3], greedy;
    struct Img {
        int first = 0, nx = 0, ny = 0;
        std::string ppm, sve;
    };
    std::vector<Img> images;
    std::string logits;
};

Case read_case(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    Case c;
    std::string line;
    const fs::path dir = fs::path(path).parent_path();
    auto rel = [&](const std::string & p) { return fs::path(p).is_absolute() ? p : (dir / p).string(); };
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string key;
        in >> key;
        std::string rest;
        std::getline(in, rest);
        if (key == "tokens") c.tokens = parse_ints(rest);
        else if (key == "pos_t") c.pos[0] = parse_ints(rest);
        else if (key == "pos_h") c.pos[1] = parse_ints(rest);
        else if (key == "pos_w") c.pos[2] = parse_ints(rest);
        else if (key == "greedy") c.greedy = parse_ints(rest);
        else if (key == "logits") c.logits = rel(std::string(rest.begin() + long(rest.find_first_not_of(' ')), rest.end()));
        else if (key == "image") {
            Case::Img im;
            std::istringstream r(rest);
            r >> im.first >> im.nx >> im.ny >> im.ppm >> im.sve;
            im.ppm = rel(im.ppm);
            im.sve = rel(im.sve);
            c.images.push_back(im);
        }
    }
    for (int a = 0; a < 3; ++a)
        if (c.pos[a].size() != c.tokens.size()) throw std::runtime_error("case: positions do not cover the tokens");
    return c;
}

std::vector<std::pair<std::int32_t, float>> top_k(const float * x, std::size_t n, int k) {
    std::vector<std::int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](std::int32_t a, std::int32_t b) { return x[a] != x[b] ? x[a] > x[b] : a < b; });
    std::vector<std::pair<std::int32_t, float>> out;
    for (int i = 0; i < k; ++i) out.push_back({idx[std::size_t(i)], x[idx[std::size_t(i)]]});
    return out;
}

std::uint64_t fnv(const void * p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char *>(p)[i]) * 1099511628211ull;
    return h;
}

void compare_logits(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    const std::size_t n = std::min(a.size(), b.size());
    double num = 0, den = 0, mx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        den += double(b[i]) * b[i];
        mx = std::max(mx, std::fabs(double(a[i]) - b[i]));
    }
    const auto ta = top_k(a.data(), n, 10), tb = top_k(b.data(), n, 10);
    int common = 0;
    for (const auto & x : ta)
        for (const auto & y : tb) common += x.first == y.first;
    std::printf("%s: rel L2 %.3e, max |diff| %.3f, top-1 %d vs %d (%s), %d of the top-10 shared\n", what, std::sqrt(num / std::max(den, 1e-30)), mx,
                ta[0].first, tb[0].first, ta[0].first == tb[0].first ? "same" : "DIFFERENT", common);
    std::printf("  ours:  ");
    for (const auto & x : ta) std::printf("%d:%.3f ", x.first, x.second);
    std::printf("\n  other: ");
    for (const auto & x : tb) std::printf("%d:%.3f ", x.first, x.second);
    std::printf("\n");
}

std::int32_t argmax(const std::vector<float> & x) { return std::int32_t(std::max_element(x.begin(), x.end()) - x.begin()); }

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0 && k != "-m" && k != "-n") throw std::runtime_error("unexpected argument " + k);
            const bool flag = k == "--cpu-ref" || k == "--llama-embd" || k == "--compare-ref" || k == "--test-snapshot" || k == "--no-graphs";
            a[k] = flag ? "1" : (i + 1 < argc ? argv[++i] : throw std::runtime_error(k + " needs a value"));
        }
        auto has = [&](const char * k) { return a.count(k) != 0; };
        if (!has("--mmproj")) throw std::runtime_error("usage: see the header of tools/flashnext/fn_vision.cpp");
        const auto t_load = clk::now();
        VisionEncoder enc(a["--mmproj"]);
        if (has("--merger-gelu")) {
            if (a["--merger-gelu"] != "erf" && a["--merger-gelu"] != "tanh") throw std::runtime_error("--merger-gelu: erf or tanh");
            enc.set_exact_merger_gelu(a["--merger-gelu"] == "erf");
        }
        std::printf("vision encoder loaded in %.0f ms: %d layers of %d, projection %d\n", ms_since(t_load), enc.config().layers,
                    enc.config().hidden, enc.config().out_dim);

        if (has("--image")) {
            const Image im = read_ppm(a["--image"]);
            int gh = 0, gw = 0;
            const std::vector<float> patches = patches_of(im, gh, gw);
            const int P = gh * gw, T = P / 4;
            const std::size_t ws = enc.workspace_bytes(P);
            void * dev = nullptr;
            if (cudaMalloc(&dev, ws) != cudaSuccess) throw std::runtime_error("cudaMalloc of the workspace failed");
            Sve ours{T, gw / 2, gh / 2, enc.config().out_dim, std::vector<float>(std::size_t(T) * enc.config().out_dim)};
            const int repeat = has("--repeat") ? std::stoi(a["--repeat"]) : 1;
            for (int r = 0; r < repeat; ++r) {
                const VisionTiming t = enc.encode(patches.data(), gh, gw, dev, ours.rows.data());
                std::printf("encode %dx%d (%d patches, %d tokens): %.1f ms wall, %.1f ms GPU, workspace %.0f MiB\n", im.w, im.h, P, T, t.total_ms,
                            t.gpu_ms, double(ws) / 1048576.0);
            }
            cudaFree(dev);
            std::printf("hash of the embeddings %016llx\n", (unsigned long long) fnv(ours.rows.data(), ours.rows.size() * sizeof(float)));
            if (has("--embd-out")) write_sve(a["--embd-out"], ours);
            Sve llama;
            if (has("--compare")) {
                llama = read_sve(a["--compare"]);
                if (llama.nx != ours.nx || llama.ny != ours.ny) std::printf("grid differs: llama %dx%d, ours %dx%d\n", llama.nx, llama.ny, ours.nx, ours.ny);
                compare("ours vs llama.cpp", ours.rows, llama.rows, ours.dim);
            }
            if (has("--cpu-ref")) {
                GgufModel g({a["--mmproj"]});
                CpuVision cpu(g);
                if (has("--merger-gelu")) cpu.exact_merger_gelu = a["--merger-gelu"] == "erf";
                const auto t0 = clk::now();
                const std::vector<float> ref = cpu.encode(patches, gh, gw);
                std::printf("cpu reference: %.0f ms\n", ms_since(t0));
                compare("ours vs cpu reference", ours.rows, ref, ours.dim);
                if (!llama.rows.empty()) compare("llama.cpp vs cpu reference", llama.rows, ref, ours.dim);
            }
        }

        if (has("--case")) {
            if (!has("-m")) throw std::runtime_error("--case needs -m <model>");
            const Case c = read_case(a["--case"]);
            const std::size_t n = c.tokens.size();
            GgufModel model(shard_paths(a["-m"]));
            EngineOptions eo;
            eo.max_ctx = has("--ctx") ? std::stoll(a["--ctx"]) : 65536;
            if (has("--mtp")) eo.mtp_path = a["--mtp"];
            if (has("--cache-mib")) eo.expert_cache_mib = std::stoll(a["--cache-mib"]);
            if (has("--routing-stats")) eo.routing_stats = a["--routing-stats"];
            if (has("--no-graphs")) eo.cuda_graphs = false;
            const auto t0 = clk::now();
            Engine engine(model, eo);
            std::printf("engine loaded in %.1f s (expert cache %.2f GiB)\n", ms_since(t0) / 1000.0, engine.stats().cache_gib);
            // image rows: ours (encoded in VRAM lent by the expert cache) or llama.cpp's
            std::vector<std::vector<float>> rows(c.images.size());
            std::vector<const float *> emb(n, nullptr);
            double encode_ms = 0;
            for (std::size_t k = 0; k < c.images.size(); ++k) {
                const Case::Img & ci = c.images[k];
                const Sve llama = read_sve(ci.sve);
                if (llama.nx != ci.nx || llama.ny != ci.ny) throw std::runtime_error("case: grid differs from the .sve");
                if (has("--llama-embd")) {
                    rows[k] = llama.rows;
                } else {
                    const Image im = read_ppm(ci.ppm);
                    int gh = 0, gw = 0;
                    const std::vector<float> patches = patches_of(im, gh, gw);
                    if (gw / 2 != ci.nx || gh / 2 != ci.ny) throw std::runtime_error("case: image grid differs from llama.cpp's (resized?)");
                    rows[k].resize(std::size_t(ci.nx) * ci.ny * enc.config().out_dim);
                    const auto e0 = clk::now();
                    void * ws = engine.lend_vram(enc.workspace_bytes(gh * gw));
                    const VisionTiming t = enc.encode(patches.data(), gh, gw, ws, rows[k].data());
                    encode_ms += ms_since(e0);
                    std::printf("image %zu: %dx%d -> %d tokens, encoded in %.1f ms (GPU %.1f ms), lent %.2f GiB\n", k, im.w, im.h, ci.nx * ci.ny,
                                t.total_ms, t.gpu_ms, engine.stats().lent_gib);
                    compare("  vs llama.cpp's rows", rows[k], llama.rows, enc.config().out_dim);
                }
                for (int i = 0; i < ci.nx * ci.ny; ++i) emb[std::size_t(ci.first + i)] = rows[k].data() + std::size_t(i) * enc.config().out_dim;
            }
            std::vector<std::int32_t> pos(3 * n);
            for (int ax = 0; ax < 3; ++ax) std::copy(c.pos[ax].begin(), c.pos[ax].end(), pos.begin() + std::ptrdiff_t(ax * n));
            // the prompt, whole or in pieces (as the server feeds long prompts)
            const std::size_t piece = has("--pieces") ? std::size_t(std::stoll(a["--pieces"])) : n;
            std::vector<float> logits;
            const auto p0 = clk::now();
            for (std::size_t i = 0; i < n; i += piece) {
                const std::size_t m = std::min(piece, n - i);
                std::vector<std::int32_t> tk(c.tokens.begin() + std::ptrdiff_t(i), c.tokens.begin() + std::ptrdiff_t(i + m));
                std::vector<std::int32_t> ps(3 * m);
                for (int ax = 0; ax < 3; ++ax) std::copy_n(pos.begin() + std::ptrdiff_t(ax * n + i), m, ps.begin() + std::ptrdiff_t(ax * m));
                ForwardInputs in;
                in.positions = ps.data();
                in.embeddings = emb.data() + i;
                logits = engine.forward(tk, false, in);
            }
            const double prompt_ms = ms_since(p0);
            std::printf("prompt of %zu tokens (%zu image tokens) in %.0f ms (%.0f tok/s), images %.0f ms; next text position %lld\n", n,
                        std::size_t(std::count_if(emb.begin(), emb.end(), [](const float * p) { return p != nullptr; })), prompt_ms,
                        double(n) / prompt_ms * 1000.0, encode_ms, (long long) engine.next_position());
            std::printf("hash of the prompt logits %016llx\n", (unsigned long long) fnv(logits.data(), logits.size() * sizeof(float)));
            if (!c.logits.empty()) {
                std::vector<float> ll(logits.size());
                std::ifstream f(c.logits, std::ios::binary);
                f.read(reinterpret_cast<char *>(ll.data()), std::streamsize(ll.size() * sizeof(float)));
                if (f) compare_logits("first logits vs llama.cpp", logits, ll);
                else std::printf("could not read %s\n", c.logits.c_str());
            }
            std::optional<EngineSnapshot> snap;
            if (has("--test-snapshot")) snap = engine.snapshot();
            // greedy continuation (with MTP drafts when loaded)
            const int n_gen = has("-n") ? std::stoi(a["-n"]) : int(std::max<std::size_t>(c.greedy.size(), 8));
            auto generate = [&](std::vector<float> lg) {
                std::vector<std::int32_t> out;
                const int k = has("--draft") ? std::stoi(a["--draft"]) : 2;
                std::int64_t drafted = 0, accepted = 0;
                const auto g0 = clk::now();
                while (int(out.size()) < n_gen) {
                    const std::int32_t next = argmax(lg);
                    out.push_back(next);
                    if (int(out.size()) >= n_gen) break;
                    if (engine.has_mtp()) {
                        const std::vector<std::int32_t> d = engine.draft(next, k);
                        std::vector<std::int32_t> step{next};
                        step.insert(step.end(), d.begin(), d.end());
                        const std::vector<float> rows_lg = engine.forward(step, true);
                        const std::size_t V = rows_lg.size() / step.size();
                        std::size_t kept = 0;
                        for (; kept < d.size(); ++kept) {
                            std::vector<float> r(rows_lg.begin() + std::ptrdiff_t(kept * V), rows_lg.begin() + std::ptrdiff_t((kept + 1) * V));
                            if (argmax(r) != d[kept] || int(out.size()) >= n_gen) break;
                            out.push_back(d[kept]);
                        }
                        drafted += std::int64_t(d.size());
                        accepted += std::int64_t(kept);
                        if (kept < d.size()) engine.rollback(int(1 + kept));
                        lg.assign(rows_lg.begin() + std::ptrdiff_t(kept * V), rows_lg.begin() + std::ptrdiff_t((kept + 1) * V));
                    } else {
                        lg = engine.forward({next});
                    }
                }
                const double gms = ms_since(g0);
                std::printf("generated %zu tokens in %.0f ms (%.1f tok/s)%s", out.size(), gms, double(out.size()) / gms * 1000.0,
                            engine.has_mtp() ? "" : "\n");
                if (engine.has_mtp()) std::printf(", MTP accepted %lld of %lld drafts\n", (long long) accepted, (long long) drafted);
                return out;
            };
            const std::vector<std::int32_t> gen = generate(logits);
            std::printf("greedy ours:  ");
            for (auto t : gen) std::printf("%d ", t);
            std::printf("\n");
            if (!c.greedy.empty()) {
                std::printf("greedy llama: ");
                for (auto t : c.greedy) std::printf("%d ", t);
                std::size_t same = 0;
                while (same < gen.size() && same < c.greedy.size() && gen[same] == c.greedy[same]) ++same;
                std::printf("\nfirst %zu greedy tokens the same\n", same);
            }
            if (snap) {
                engine.restore(*snap);
                const std::vector<std::int32_t> again = generate(logits);
                std::printf("after restore: %s\n", again == gen ? "the same continuation" : "DIFFERENT continuation");
            }
            if (has("--compare-ref")) {
                ReferenceModel ref(model);
                const auto r0 = clk::now();
                const std::vector<float> rl = ref.forward(c.tokens, false, pos.data(), emb.data());
                std::printf("FP32 reference prompt in %.0f s\n", ms_since(r0) / 1000.0);
                compare_logits("first logits vs the FP32 reference", logits, rl);
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fn_vision: %s\n", e.what());
        return 1;
    }
}
