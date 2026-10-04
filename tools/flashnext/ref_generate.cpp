// Greedy generation with the FP32 reference forward pass of Qwen3.8-Flash-Next.
//
// Usage: ref_generate -m <shard 1 of the GGUF> (--tokens 1,2,3 | --tokens-file ids.txt) [-n 32]
//                     [--threads N] [--chunk N] [--json out.json] [--dump dir [--dump-filter regex]] [--force dir]
//                     [--emulate-llamacpp cuda|cpu] [--self-test] [--consistency]
//
// Prints the generated ids, the top-5 logits (and log-probabilities) at every step, and timings.
// The other shards of a split GGUF are found next to the first one. --self-test checks the batched
// matmul against a double-precision loop; --consistency checks that a batched prefill and
// token-by-token decoding give the same logits. --dump writes every named intermediate of the prompt
// pass to <dir>/<name>-<layer>.bin ([int32 width, n_tokens, 1, 1] + floats; a name repeated within a
// layer gets #1, #2, ... as in llama.cpp's graph callback). --emulate-llamacpp
// rounds intermediates like llama.cpp on CUDA or on the CPU (diagnostic only). --force <dir> replaces
// every live intermediate of the prompt pass by the same-named tensor of a llama.cpp graph dump (see
// llama.cpp's cb_eval names; [int32 ne0..ne3] + floats), so each component is checked on llama.cpp's
// own inputs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashnext/gguf.h"
#include "flashnext/reference.h"

using namespace ninfer::flashnext;
namespace fs = std::filesystem;

namespace {

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

std::vector<std::int32_t> parse_ids(const std::string & text) {
    std::vector<std::int32_t> ids;
    std::string cur;
    for (char ch : text + ",") {
        if ((ch >= '0' && ch <= '9') || ch == '-') {
            cur += ch;
        } else if (!cur.empty()) {
            ids.push_back(std::int32_t(std::stol(cur)));
            cur.clear();
        }
    }
    return ids;
}

struct Top {
    std::int32_t id;
    float logit;
    double logprob;
};

std::vector<Top> top_k(const std::vector<float> & logits, int k) {
    std::vector<std::int32_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](std::int32_t a, std::int32_t b) {
        return logits[std::size_t(a)] != logits[std::size_t(b)] ? logits[std::size_t(a)] > logits[std::size_t(b)] : a < b;
    });
    const float mx = logits[std::size_t(idx[0])];
    double sum = 0.0;
    for (float v : logits) sum += std::exp(double(v) - double(mx));
    const double lse = double(mx) + std::log(sum);
    std::vector<Top> out;
    for (int i = 0; i < k; ++i) out.push_back({idx[std::size_t(i)], logits[std::size_t(idx[std::size_t(i)])], double(logits[std::size_t(idx[std::size_t(i)])]) - lse});
    return out;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model_path, tokens_arg, tokens_file, json_path, dump_dir, force_dir, dump_filter;
    int n_gen = 32;
    ReferenceOptions opt;
    bool self_test = false, consistency = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--tokens") tokens_arg = next();
        else if (a == "--tokens-file") tokens_file = next();
        else if (a == "-n") n_gen = std::stoi(next());
        else if (a == "--threads") opt.n_threads = std::stoi(next());
        else if (a == "--chunk") opt.max_chunk = std::stoll(next());
        else if (a == "--json") json_path = next();
        else if (a == "--dump") dump_dir = next();
        else if (a == "--force") force_dir = next();
        else if (a == "--dump-filter") dump_filter = next();
        else if (a == "--emulate-llamacpp") {
            const std::string v = next();
            if (v == "cuda") opt.emulate = ReferenceOptions::Emulate::llamacpp_cuda;
            else if (v == "cpu") opt.emulate = ReferenceOptions::Emulate::llamacpp_cpu;
            else throw std::runtime_error("--emulate-llamacpp takes cuda or cpu");
        }
        else if (a == "--self-test") self_test = true;
        else if (a == "--consistency") consistency = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: ref_generate -m <gguf shard 1> (--tokens 1,2,3 | --tokens-file f) [-n 32] [--threads N]\n"
                             "                    [--chunk N] [--json out.json] [--dump dir [--dump-filter regex]] [--force dir]\n"
                             "                    [--emulate-llamacpp cuda|cpu] [--self-test] [--consistency]\n");
        return 2;
    }
    std::vector<std::int32_t> prompt;
    if (!tokens_file.empty()) {
        std::ifstream f(tokens_file);
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = parse_ids(ss.str());
    } else {
        prompt = parse_ids(tokens_arg);
    }

    auto t_load = std::chrono::steady_clock::now();
    GgufModel gguf(shard_paths(model_path));
    ReferenceModel model(gguf, opt);
    const ReferenceConfig & cfg = model.config();
    std::printf("loaded %s: %d layers, n_embd %d, vocab %d, in %.1f s\n", model_path.c_str(), cfg.n_layer, cfg.n_embd,
                cfg.n_vocab, seconds_since(t_load));

    if (self_test) {
        const double err = model.self_test_matmul();
        std::printf("matmul self-test: worst error relative to sum|w*x| = %.3g %s\n", err, err < 1e-5 ? "OK" : "FAILED");
        if (err >= 1e-5) return 1;
    }
    if (prompt.empty()) return 0;

    if (consistency) {
        // batched prefill vs token by token: same last-token logits
        model.reset();
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> a = model.forward(prompt);
        const double ta = seconds_since(t0);
        model.reset();
        t0 = std::chrono::steady_clock::now();
        std::vector<float> b;
        for (std::int32_t tok : prompt) b = model.forward({tok});
        const double tb = seconds_since(t0);
        double max_diff = 0.0, max_abs = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            max_diff = std::max(max_diff, double(std::fabs(a[i] - b[i])));
            max_abs = std::max(max_abs, double(std::fabs(a[i])));
        }
        std::printf("consistency: batched (%.1f s) vs token-by-token (%.1f s) over %zu tokens: max |dlogit| = %.3g (max |logit| %.3g)\n",
                    ta, tb, prompt.size(), max_diff, max_abs);
        model.reset();
    }

    std::printf("prompt: %zu tokens%s\n", prompt.size(), opt.emulate != ReferenceOptions::Emulate::none ? " (emulating llama.cpp rounding)" : "");
    if (!dump_dir.empty()) {
        fs::create_directories(dump_dir);
        model.set_activation_hook([&, seen = std::map<std::string, int>()](const std::string & name, int layer, std::int64_t,
                                                                         std::int64_t n_tokens, std::int64_t width,
                                                                         const float * data) mutable {
            // a name repeated within a layer gets #1, #2, ... as llama.cpp's graph callback names them
            std::string base = name + (layer >= 0 ? "-" + std::to_string(layer) : "");
            const int k = seen[base]++;
            if (k > 0) base += "#" + std::to_string(k);
            if (!dump_filter.empty() && !std::regex_match(base, std::regex(dump_filter))) return;
            const std::string file = dump_dir + "/" + base + ".bin";
            std::ofstream f(file, std::ios::binary);
            const std::int32_t ne[4] = {std::int32_t(width), std::int32_t(n_tokens), 1, 1};
            f.write(reinterpret_cast<const char *>(ne), sizeof(ne));
            f.write(reinterpret_cast<const char *>(data), std::streamsize(std::size_t(width * n_tokens) * sizeof(float)));
        });
    }
    std::vector<std::int32_t> generated;
    std::vector<std::vector<Top>> tops;
    std::int64_t n_forced = 0;
    if (!force_dir.empty()) {
        model.set_activation_override([&, seen = std::map<std::string, int>()](const std::string & name, int layer, std::int64_t,
                                                                             std::int64_t n_tokens, std::int64_t width,
                                                                             float * data) mutable {
            std::string base = name + (layer >= 0 ? "-" + std::to_string(layer) : "");
            const int k = seen[base]++;
            if (k > 0) base += "#" + std::to_string(k);
            // our names -> llama.cpp's graph names
            const std::string L = std::to_string(layer);
            std::string theirs = base;
            if (base == "hc_attn_mixed-" + L) theirs = "hc_mixed-" + L;
            else if (base == "hc_ffn_mixed-" + L) theirs = "hc_mixed-" + L + "#1";
            else if (base == "hc_combine-" + L + "#1" || base == "l_out-" + L) theirs = "l_last-" + L;
            else if (name == "ple_embd") theirs = "ple_embd";
            else if (base == "Kcur-" + L || base == "Vcur-" + L) theirs = base + "#1";  // after norm and rope
            std::ifstream f(force_dir + "/" + theirs + ".bin", std::ios::binary);
            if (!f) return false;
            std::int32_t ne[4];
            f.read(reinterpret_cast<char *>(ne), sizeof(ne));
            const std::int64_t n = std::int64_t(ne[0]) * ne[1] * ne[2] * ne[3];
            if (n != width * n_tokens) return false;
            f.read(reinterpret_cast<char *>(data), std::streamsize(n * std::int64_t(sizeof(float))));
            ++n_forced;
            return true;
        });
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = model.forward(prompt);
    if (!force_dir.empty()) {
        std::printf("forced %lld intermediates from %s\n", (long long) n_forced, force_dir.c_str());
        model.set_activation_override(nullptr);
    }
    const double t_prefill = seconds_since(t0);
    model.set_activation_hook(nullptr);
    std::printf("prefill: %.2f s (%.2f tok/s)\n", t_prefill, double(prompt.size()) / t_prefill);
    std::vector<double> step_times;
    for (int step = 0; step < n_gen; ++step) {
        std::vector<Top> top = top_k(logits, 5);
        const std::int32_t next = top[0].id;
        generated.push_back(next);
        tops.push_back(top);
        std::printf("step %2d: id %6d |", step, next);
        for (const Top & tp : top) std::printf(" %6d %8.4f (%7.4f)", tp.id, tp.logit, tp.logprob);
        std::printf("\n");
        if (step + 1 == n_gen) break;
        auto ts = std::chrono::steady_clock::now();
        logits = model.forward({next});
        step_times.push_back(seconds_since(ts));
    }
    std::printf("generated:");
    for (std::int32_t id : generated) std::printf(" %d", id);
    std::printf("\n");
    if (!step_times.empty()) {
        const double total = std::accumulate(step_times.begin(), step_times.end(), 0.0);
        std::printf("decode: %zu tokens, %.3f s/token (%.2f tok/s)\n", step_times.size(), total / double(step_times.size()),
                    double(step_times.size()) / total);
    }

    if (!json_path.empty()) {
        std::ofstream f(json_path);
        f << "{\n  \"prompt_tokens\": " << prompt.size() << ",\n  \"prefill_s\": " << t_prefill << ",\n  \"tokens\": [";
        for (std::size_t i = 0; i < generated.size(); ++i) f << (i ? ", " : "") << generated[i];
        f << "],\n  \"step_s\": [";
        for (std::size_t i = 0; i < step_times.size(); ++i) f << (i ? ", " : "") << step_times[i];
        f << "],\n  \"top\": [\n";
        for (std::size_t s = 0; s < tops.size(); ++s) {
            f << "    [";
            for (std::size_t j = 0; j < tops[s].size(); ++j) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "{\"id\": %d, \"logit\": %.6f, \"logprob\": %.6f}", tops[s][j].id, tops[s][j].logit,
                              tops[s][j].logprob);
                f << (j ? ", " : "") << buf;
            }
            f << "]" << (s + 1 < tops.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
    }
    return 0;
}
