// Greedy generation with the Flash-Next GPU engine, optionally checked against the FP32 reference.
//
// Usage: fn_generate -m <shard 1 of the GGUF> (--tokens 1,2,3 | --tokens-file ids.txt) [-n 32] [--ctx N]
//                    [--threads N] [--json out.json] [--dump dir] [--compare-ref]
//                    [--cache-mib N] [--reserve-mib N] [--routing-stats file] [--no-graphs] [--prefill-chunk N]
//                    [--no-host-images] [--gpu-miss-permille N] [--test-snapshot] [--mtp mtp.gguf [--draft K]]
//
// Prints the generated ids, the top-5 logits at every step, and prefill/decode speed. The JSON has the
// same layout as ref_generate's. --dump writes the prompt pass's intermediates like ref_generate does.
// --compare-ref also runs the reference on the prompt in this process and prints, per layer, the
// relative error of every intermediate both produce, then the agreement of the prompt's logits.
// --routing-stats loads expert routing counts to choose the experts kept in VRAM, and saves the
// updated counts at the end.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "flashnext/engine.h"
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

std::vector<Top> top_k(const float * logits, std::size_t n, int k) {
    std::vector<std::int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](std::int32_t a, std::int32_t b) {
        return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
    });
    const float mx = logits[idx[0]];
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += std::exp(double(logits[i]) - double(mx));
    const double lse = double(mx) + std::log(sum);
    std::vector<Top> out;
    for (int i = 0; i < k; ++i) out.push_back({idx[std::size_t(i)], logits[idx[std::size_t(i)]], double(logits[idx[std::size_t(i)]]) - lse});
    return out;
}

std::int32_t argmax(const float * x, std::size_t n) {
    return std::int32_t(std::max_element(x, x + n) - x);  // first of equal maxima, like top_k
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Intermediates of one prompt pass: (layer, name) -> rows appended across calls.
struct Capture {
    std::map<std::pair<int, std::string>, std::pair<std::int64_t, std::vector<float>>> rows;  // width, data
    std::vector<std::pair<int, std::string>> order;
    void add(const std::string & name, int layer, std::int64_t n_tokens, std::int64_t width, const float * data) {
        const auto key = std::make_pair(layer, name);
        auto it = rows.find(key);
        if (it == rows.end()) {
            order.push_back(key);
            it = rows.emplace(key, std::make_pair(width, std::vector<float>())).first;
        }
        it->second.second.insert(it->second.second.end(), data, data + n_tokens * width);
    }
};

}  // namespace

static int run(int argc, char ** argv);

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int run(int argc, char ** argv) {
    std::string model_path, tokens_arg, tokens_file, json_path, dump_dir;
    int n_gen = 16;
    bool compare_ref = false, test_snapshot = false;
    int n_draft = 2;
    EngineOptions opt;
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
        else if (a == "--ctx") opt.max_ctx = std::stoll(next());
        else if (a == "--threads") opt.cpu_threads = std::stoi(next());
        else if (a == "--json") json_path = next();
        else if (a == "--dump") dump_dir = next();
        else if (a == "--compare-ref") compare_ref = true;
        else if (a == "--cache-mib") opt.expert_cache_mib = std::stoll(next());
        else if (a == "--reserve-mib") opt.vram_reserve_mib = std::stoll(next());
        else if (a == "--routing-stats") opt.routing_stats = next();
        else if (a == "--no-graphs") opt.cuda_graphs = false;
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--test-snapshot") test_snapshot = true;
        else if (a == "--no-host-images") opt.host_expert_images = false;
        else if (a == "--mtp") opt.mtp_path = next();
        else if (a == "--int8-cpu-experts") opt.precise_cpu_experts = false;
        else if (a == "--no-pin") opt.pin_cpu_threads = false;
        else if (a == "--draft") n_draft = std::stoi(next());
        else if (a == "--gpu-miss-permille") opt.gpu_miss_permille = std::stoi(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (model_path.empty() || (tokens_arg.empty() && tokens_file.empty())) {
        std::fprintf(stderr, "usage: fn_generate -m <gguf shard 1> (--tokens 1,2,3 | --tokens-file f) [-n 32] [--ctx N] [--threads N]\n"
                             "                   [--json out.json] [--dump dir] [--compare-ref]\n");
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
    if (prompt.empty()) throw std::runtime_error("empty prompt");

    auto t_load = std::chrono::steady_clock::now();
    GgufModel gguf(shard_paths(model_path));
    Engine engine(gguf, opt);
    std::printf("loaded %s in %.1f s; expert cache: %lld experts in %.2f GiB of VRAM\n", model_path.c_str(), seconds_since(t_load),
                (long long) engine.stats().cached_experts, engine.stats().cache_gib);

    Capture mine;
    const bool capture = compare_ref || !dump_dir.empty();
    if (capture) {
        engine.set_activation_hook([&](const std::string & name, int layer, std::int64_t, std::int64_t n_tokens, std::int64_t width,
                                       const float * data) { mine.add(name, layer, n_tokens, width, data); });
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = engine.forward(prompt);
    const double t_prefill = seconds_since(t0);
    engine.set_activation_hook(nullptr);
    std::printf("prompt: %zu tokens, prefill %.2f s (%.1f tok/s)%s; %lld cached experts swapped (%.2f s); CPU experts %.2f s\n", prompt.size(),
                t_prefill, double(prompt.size()) / t_prefill, capture ? " with intermediates captured" : "", (long long) engine.stats().cache_swaps,
                engine.stats().cache_swap_ms / 1e3, engine.stats().cpu_experts_ms / 1e3);

    if (!dump_dir.empty()) {
        fs::create_directories(dump_dir);
        for (const auto & key : mine.order) {
            const auto & [width, data] = mine.rows[key];
            const std::string file = dump_dir + "/" + key.second + (key.first >= 0 ? "-" + std::to_string(key.first) : "") + ".bin";
            std::ofstream f(file, std::ios::binary);
            const std::int32_t ne[4] = {std::int32_t(width), std::int32_t(std::int64_t(data.size()) / width), 1, 1};
            f.write(reinterpret_cast<const char *>(ne), sizeof(ne));
            f.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * sizeof(float)));
        }
        std::printf("dumped %zu intermediates to %s\n", mine.order.size(), dump_dir.c_str());
    }

    if (compare_ref) {
        Capture ref;
        ReferenceModel reference(gguf);
        reference.set_activation_hook([&](const std::string & name, int layer, std::int64_t, std::int64_t n_tokens, std::int64_t width,
                                          const float * data) { ref.add(name, layer, n_tokens, width, data); });
        auto tr = std::chrono::steady_clock::now();
        const std::vector<float> ref_logits = reference.forward(prompt);
        std::printf("reference prompt pass: %.1f s\n", seconds_since(tr));
        // per layer: the relative error of each intermediate both produced
        int last_layer = -2;
        for (const auto & key : mine.order) {
            auto it = ref.rows.find(key);
            if (it == ref.rows.end()) continue;
            const auto & a = mine.rows[key].second;
            const auto & b = it->second.second;
            if (a.size() != b.size() || mine.rows[key].first != it->second.first) {
                std::printf("  %s-%d: shape differs (%zu vs %zu)\n", key.second.c_str(), key.first, a.size(), b.size());
                continue;
            }
            double num = 0, den = 0;
            if (key.second == "ffn_moe_topk") {
                // fraction of selected experts that differ, per token as a set
                const std::int64_t w = mine.rows[key].first;
                std::int64_t diff = 0;
                for (std::size_t r = 0; r * w < a.size(); ++r) {
                    std::vector<float> x(a.begin() + r * w, a.begin() + (r + 1) * w), y(b.begin() + r * w, b.begin() + (r + 1) * w);
                    std::sort(x.begin(), x.end());
                    std::sort(y.begin(), y.end());
                    std::vector<float> common;
                    std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(common));
                    diff += w - std::int64_t(common.size());
                }
                num = double(diff);
                den = double(a.size());
            } else {
                for (std::size_t i = 0; i < a.size(); ++i) {
                    num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
                    den += double(b[i]) * b[i];
                }
                num = std::sqrt(num);
                den = std::sqrt(den);
            }
            if (key.first != last_layer) {
                std::printf("%slayer %3d:", last_layer == -2 ? "" : "\n", key.first);
                last_layer = key.first;
            }
            std::printf(" %s %.2e", key.second.c_str(), den > 0 ? num / den : num);
        }
        std::printf("\n");
        const std::size_t V = std::size_t(engine.n_vocab());
        double num = 0, den = 0, maxd = 0;
        for (std::size_t i = 0; i < V; ++i) {
            const double d = double(logits[i]) - ref_logits[i];
            num += d * d;
            den += double(ref_logits[i]) * ref_logits[i];
            maxd = std::max(maxd, std::fabs(d));
        }
        const auto ta = top_k(logits.data(), V, 5), tb = top_k(ref_logits.data(), V, 5);
        std::printf("prompt logits vs reference: rel %.3e, max |d| %.4f; top-1 %d vs %d (%s); top-5 logprobs mine/ref:", std::sqrt(num / den),
                    maxd, ta[0].id, tb[0].id, ta[0].id == tb[0].id ? "same" : "DIFFERENT");
        for (int j = 0; j < 5; ++j) std::printf(" %d %.3f/%.3f", ta[std::size_t(j)].id, ta[std::size_t(j)].logprob, tb[std::size_t(j)].logprob);
        std::printf("\n");
    }

    if (test_snapshot) {
        // snapshot after the prompt, continue greedily, return to the snapshot, continue again
        auto t_snap = std::chrono::steady_clock::now();
        const EngineSnapshot snap = engine.snapshot();
        const double snap_ms = 1e3 * seconds_since(t_snap);
        auto greedy = [&](std::vector<float> lg) {
            std::vector<std::int32_t> ids;
            std::vector<float> last;
            for (int i = 0; i < 8; ++i) {
                const std::int32_t id = top_k(lg.data(), std::size_t(engine.n_vocab()), 1)[0].id;
                ids.push_back(id);
                lg = engine.forward({id});
            }
            return std::make_pair(ids, lg);
        };
        const auto first = greedy(logits);
        t_snap = std::chrono::steady_clock::now();
        engine.restore(snap);
        const double restore_ms = 1e3 * seconds_since(t_snap);
        const auto second = greedy(logits);
        const bool same = first.first == second.first && first.second == second.second;
        std::printf("snapshot %.1f MiB in %.1f ms, restore in %.1f ms; continuation after restore identical: %s\n",
                    snap.state.size() / 1048576.0, snap_ms, restore_ms, same ? "yes" : "NO");
        engine.restore(snap);
        if (!same) return 1;
    }

    std::vector<std::int32_t> generated;
    std::vector<std::vector<Top>> tops;
    std::vector<double> step_times;
    const EngineStats after_prompt = engine.stats();
    const std::size_t V = std::size_t(engine.n_vocab());
    if (engine.has_mtp() && n_draft > 0) {
        // greedy speculative decoding: draft with the MTP head, verify in one step, keep the agreed prefix
        std::int32_t tok = top_k(logits.data(), V, 1)[0].id;
        int steps = 0, drafted = 0, accepted = 0;
        double draft_s = 0, verify_s = 0;
        const double cpu0 = engine.stats().cpu_experts_ms;
        auto ts = std::chrono::steady_clock::now();
        while (int(generated.size()) < n_gen) {
            generated.push_back(tok);
            if (int(generated.size()) == n_gen) break;
            std::vector<std::int32_t> seq{tok};
            auto td = std::chrono::steady_clock::now();
            const std::vector<std::int32_t> d = engine.draft(tok, n_draft);
            draft_s += seconds_since(td);
            seq.insert(seq.end(), d.begin(), d.end());
            auto tv = std::chrono::steady_clock::now();
            const std::vector<float> lg = engine.forward(seq, true);
            verify_s += seconds_since(tv);
            int keep = 1;
            std::int32_t next = argmax(lg.data(), V);
            for (std::size_t i = 0; i < d.size() && next == d[i]; ++i) {
                generated.push_back(d[i]);
                ++keep;
                next = argmax(lg.data() + std::size_t(keep - 1) * V, V);
            }
            engine.rollback(keep);
            ++steps;
            drafted += int(d.size());
            accepted += keep - 1;
            tok = next;
        }
        const double total = seconds_since(ts);
        if (int(generated.size()) > n_gen) generated.resize(std::size_t(n_gen));
        std::printf("generated:");
        for (std::int32_t id : generated) std::printf(" %d", id);
        std::printf("\nMTP decode: %zu tokens in %d steps, %.2f ms/token (%.2f tok/s); drafts accepted %d/%d (%.1f%%), %.2f tokens per step\n",
                    generated.size(), steps, 1e3 * total / double(generated.size()), double(generated.size()) / total, accepted, drafted,
                    100.0 * accepted / std::max(1, drafted), double(generated.size()) / std::max(1, steps));
        std::printf("per step: draft %.2f ms, verify %.2f ms (CPU experts %.2f ms), other %.2f ms\n", 1e3 * draft_s / steps, 1e3 * verify_s / steps,
                    (engine.stats().cpu_experts_ms - cpu0) / steps, 1e3 * (total - draft_s - verify_s) / steps);
        if (!opt.routing_stats.empty()) engine.save_routing_stats(opt.routing_stats);
        if (!json_path.empty()) {
            std::ofstream f(json_path);
            f << "{\n  \"prompt_tokens\": " << prompt.size() << ",\n  \"tokens\": [";
            for (std::size_t i = 0; i < generated.size(); ++i) f << (i ? ", " : "") << generated[i];
            f << "]\n}\n";
        }
        return 0;
    }
    for (int step = 0; step < n_gen; ++step) {
        std::vector<Top> top = top_k(logits.data(), V, 5);
        const std::int32_t next = top[0].id;
        generated.push_back(next);
        tops.push_back(top);
        std::printf("step %2d: id %6d |", step, next);
        for (const Top & tp : top) std::printf(" %6d %8.4f (%7.4f)", tp.id, tp.logit, tp.logprob);
        std::printf("\n");
        if (step + 1 == n_gen) break;
        auto ts = std::chrono::steady_clock::now();
        logits = engine.forward({next});
        step_times.push_back(seconds_since(ts));
    }
    std::printf("generated:");
    for (std::int32_t id : generated) std::printf(" %d", id);
    std::printf("\n");
    if (!step_times.empty()) {
        const double total = std::accumulate(step_times.begin(), step_times.end(), 0.0);
        const EngineStats & st = engine.stats();
        const double cpu_ms = st.cpu_experts_ms - after_prompt.cpu_experts_ms, eng_ms = st.step_ms - after_prompt.step_ms;
        const std::int64_t hits = st.expert_hits - after_prompt.expert_hits, pairs = st.expert_pairs - after_prompt.expert_pairs;
        const std::int64_t host = st.expert_host_reads - after_prompt.expert_host_reads;
        std::printf("decode: %zu tokens, %.2f ms/token (%.2f tok/s); CPU experts %.1f%% of engine time; experts: %.1f%% VRAM, %.1f%% GPU from "
                    "host memory, %.1f%% CPU\n",
                    step_times.size(), 1e3 * total / double(step_times.size()), double(step_times.size()) / total, 100.0 * cpu_ms / eng_ms,
                    100.0 * double(hits) / double(std::max<std::int64_t>(1, pairs)), 100.0 * double(host) / double(std::max<std::int64_t>(1, pairs)),
                    100.0 * double(pairs - hits - host) / double(std::max<std::int64_t>(1, pairs)));
    }
    if (!opt.routing_stats.empty()) engine.save_routing_stats(opt.routing_stats);

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
