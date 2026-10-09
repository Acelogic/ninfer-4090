// Greedy generation with the Flash-Next GPU engine, optionally checked against the FP32 reference.
//
// Usage: fn_generate -m <shard 1 of the GGUF> (--tokens 1,2,3 | --tokens-file ids.txt) [-n 32] [--ctx N]
//                    [--threads N] [--json out.json] [--dump dir] [--compare-ref]
//                    [--cache-mib N] [--reserve-mib N] [--no-vram-budget] [--routing-stats file] [--no-graphs] [--prefill-chunk N]
//                    [--no-host-images] [--gpu-miss-permille N] [--test-snapshot [--snapshot-detour N]] [--test-park] [--mtp mtp.gguf [--draft K]
//                    [--mtp-experts-vram]] [--no-decode-adapt] [--hash]
//                    [--kv-stream 0|1] [--kv-resident CELLS] [--kv-stage-cells CELLS] [--kv-group-tokens N] [--followup N[,N...]]
//                    [--no-lend] [--no-stream] [--stream-min N] [--chunk-max N] [--cpu-share-max N] [--dense-sgemm] [--profile]
//                    [--hash-state] [--pieces N] [--prefill-runs 0,2048:nostream,...]
//
// Prints the generated ids, the top-5 logits at every step, and prefill/decode speed. The JSON has the
// same layout as ref_generate's. --dump writes the prompt pass's intermediates like ref_generate does.
// --compare-ref also runs the reference on the prompt in this process and prints, per layer, the
// relative error of every intermediate both produce, then the agreement of the prompt's logits.
// --routing-stats loads expert routing counts to choose the experts kept in VRAM, and saves the
// updated counts at the end. --prefill-chunk 0 (the default) sizes prompt chunks automatically. --hash-state
// prints hashes of the recurrent state and of the logits after the prompt (bitwise comparisons between
// runs). --prefill-runs repeats the prompt (after a reset) once per entry, chunk[:nostream], and reports each
// (the expert cache keeps what earlier runs taught it).
// --test-snapshot continues 8 tokens past the prompt, returns to the snapshot and checks that the same 8
// tokens follow; --snapshot-detour N also feeds N more tokens (the prompt's first ones) before returning, so
// that ring buffers (the MTP layer's K/V) wrap past the snapshot's positions. --test-park (implies --test-snapshot)
// parks the prompt's positions before the return instead, then overwrites every one of them (and 64 more) with
// another sequence from an empty one (the prompt reversed), unparks and restores: the same 8 tokens must follow.
// --kv-stream forces KV streaming off or on (default: on when --ctx exceeds --kv-resident, 32768 cells);
// the KV counters are printed after the prompt and at the end. --followup feeds short prompts of N tokens
// (the prompt's first ones) after the prompt and times them, like a conversation's next turn at that depth;
// decoding then continues after them.
// --hash prints 64-bit FNV-1a hashes of the prompt's logits, of every later forward()'s logits and of
// the MTP drafts, so that two builds or configurations can be checked for bitwise identical results.
#include <algorithm>
#include <chrono>
#include <cstdint>
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

// FNV-1a over raw bytes, chained through h
std::uint64_t fnv1a(const void * data, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
    const unsigned char * p = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

void print_kv(const Engine & engine, const char * when) {
    const KvStreamStats k = engine.kv_stream_stats();
    if (!k.enabled) return;
    std::printf("kv stream %s: %lld layers, %lld resident cells each (%.2f GiB VRAM, %.2f GiB pinned RAM); %llu resolves, %llu page lookups, "
                "%llu misses (%.2f%%), prompt chunks beyond it: %llu staged (%.2f GiB), %llu in groups%s\n",
                when, (long long) k.layers, (long long) k.resident_cells, k.vram_gib, k.host_gib, (unsigned long long) k.resolves,
                (unsigned long long) k.lookups, (unsigned long long) k.misses, 100.0 * double(k.misses) / double(k.lookups ? k.lookups : 1),
                (unsigned long long) k.staged_chunks, k.staged_gib, (unsigned long long) k.grouped_chunks, k.overflow ? "; OVERFLOW" : "");
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
    bool compare_ref = false, test_snapshot = false, test_park = false, hash = false, hash_state = false;
    std::uint64_t h_decode = fnv1a(nullptr, 0), h_drafts = fnv1a(nullptr, 0);
    int n_draft = 2, detour = 0;
    std::vector<std::int32_t> followups;
    std::string prefill_runs;
    std::size_t pieces = 0;  // feed the prompt in forward() calls of this many tokens (as the server does), with a snapshot after each
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
        else if (a == "--no-vram-budget") opt.vram_follow_budget = false;
        else if (a == "--routing-stats") opt.routing_stats = next();
        else if (a == "--no-graphs") opt.cuda_graphs = false;
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--test-snapshot") test_snapshot = true;
        else if (a == "--test-park") test_snapshot = test_park = true;
        else if (a == "--snapshot-detour") detour = std::stoi(next());
        else if (a == "--no-host-images") opt.host_expert_images = false;
        else if (a == "--mtp") opt.mtp_path = next();
        else if (a == "--mtp-experts-vram") opt.mtp_experts_vram = true;
        else if (a == "--no-decode-adapt") opt.decode_adapt = false;
        else if (a == "--int8-cpu-experts") opt.precise_cpu_experts = false;
        else if (a == "--pin") opt.pin_cpu_threads = true;
        else if (a == "--draft") n_draft = std::stoi(next());
        else if (a == "--gpu-miss-permille") opt.gpu_miss_permille = std::stoi(next());
        else if (a == "--no-lend") opt.prefill_lend = false;
        else if (a == "--no-stream") opt.prefill_stream = false;
        else if (a == "--stream-min") opt.prefill_stream_min = std::stoi(next());
        else if (a == "--chunk-max") opt.prefill_chunk_max = std::stoi(next());
        else if (a == "--cpu-share-max") opt.prefill_cpu_share_max = std::stoi(next());
        else if (a == "--dense-tc") opt.prefill_dense_tc = true;
        else if (a == "--dense-sgemm") opt.prefill_dense_tc = false;
        else if (a == "--profile") opt.profile = true;
        else if (a == "--hash-state") hash_state = true;
        else if (a == "--prefill-runs") prefill_runs = next();
        else if (a == "--pieces") pieces = std::stoul(next());
        else if (a == "--hash") hash = true;
        else if (a == "--kv-stream") opt.kv_stream = std::stoi(next());
        else if (a == "--kv-resident") opt.kv_resident = std::stoll(next());
        else if (a == "--kv-stage-cells") opt.kv_stage_cells = std::stoll(next());
        else if (a == "--kv-group-tokens") opt.kv_group_tokens = std::stoi(next());
        else if (a == "--followup") followups = parse_ids(next());
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
    if (engine.stats().vram_tracked)
        std::printf("VRAM budget %.2f GiB, %.2f GiB in use; %.2f GiB of the expert cache given back to keep %lld MiB free\n",
                    engine.stats().vram_budget_gib, engine.stats().vram_usage_gib, engine.stats().cache_released_gib, (long long) opt.vram_reserve_mib);

    Capture mine;
    const bool capture = compare_ref || !dump_dir.empty();
    if (capture) {
        engine.set_activation_hook([&](const std::string & name, int layer, std::int64_t, std::int64_t n_tokens, std::int64_t width,
                                       const float * data) { mine.add(name, layer, n_tokens, width, data); });
    }
    if (!prefill_runs.empty()) {
        // the same prompt once per entry "chunk[:nostream]", after a reset
        std::stringstream ss(prefill_runs);
        for (std::string item; std::getline(ss, item, ',');) {
            const bool nostream = item.find(":nostream") != std::string::npos;
            const int chunk = std::stoi(item.substr(0, item.find(':')));
            engine.set_prefill(chunk, !nostream && opt.prefill_stream);
            engine.reset();
            const EngineStats s0 = engine.stats();
            auto tr = std::chrono::steady_clock::now();
            const std::vector<float> lg = engine.forward(prompt);
            const double dt = seconds_since(tr);
            const EngineStats & s1 = engine.stats();
            std::printf("prefill run chunk %s: %zu tokens in %.2f s (%.1f tok/s); chunks of %d, %lld streamed of %lld; lent %.2f GiB; streamed %lld experts "
                        "(%.1f GiB); refill %.0f ms; swaps %lld (%.0f ms); CPU experts %.2f s; logits hash %016llx\n",
                        item.c_str(), prompt.size(), dt, double(prompt.size()) / dt, s1.last_chunk, (long long) (s1.streamed_chunks - s0.streamed_chunks),
                        (long long) (s1.prompt_chunks - s0.prompt_chunks), s1.lent_gib, (long long) (s1.streamed_experts - s0.streamed_experts),
                        s1.streamed_gib - s0.streamed_gib, s1.refill_ms - s0.refill_ms, (long long) (s1.cache_swaps - s0.cache_swaps),
                        s1.cache_swap_ms - s0.cache_swap_ms, (s1.cpu_experts_ms - s0.cpu_experts_ms) / 1e3,
                        (unsigned long long) fnv1a(lg.data(), lg.size() * sizeof(float)));
        }
        engine.reset();
        engine.set_prefill(opt.prefill_chunk, opt.prefill_stream);
    }
    const EngineStats s0 = engine.stats();
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    if (pieces) {
        for (std::size_t i = 0; i < prompt.size(); i += pieces) {
            const std::size_t e = std::min(prompt.size(), i + pieces);
            logits = engine.forward(std::vector<std::int32_t>(prompt.begin() + std::ptrdiff_t(i), prompt.begin() + std::ptrdiff_t(e)));
            if (e < prompt.size()) (void) engine.snapshot();  // the server snapshots at prefix frontiers
        }
    } else {
        logits = engine.forward(prompt);
    }
    const double t_prefill = seconds_since(t0);
    engine.set_activation_hook(nullptr);
    {
        const EngineStats & s1 = engine.stats();
        std::printf("prompt: %zu tokens, prefill %.2f s (%.1f tok/s)%s; %lld cached experts swapped (%.2f s); CPU experts %.2f s\n", prompt.size(),
                    t_prefill, double(prompt.size()) / t_prefill, capture ? " with intermediates captured" : "",
                    (long long) (s1.cache_swaps - s0.cache_swaps), (s1.cache_swap_ms - s0.cache_swap_ms) / 1e3, (s1.cpu_experts_ms - s0.cpu_experts_ms) / 1e3);
        std::printf("prefill: chunks of %d tokens, %lld chunks (%lld streamed); lent %.2f GiB; streamed %lld experts (%.1f GiB), %lld on the CPU; "
                    "refill %.0f ms; pinned layers %d\n",
                    s1.last_chunk, (long long) (s1.prompt_chunks - s0.prompt_chunks), (long long) (s1.streamed_chunks - s0.streamed_chunks), s1.lent_gib,
                    (long long) (s1.streamed_experts - s0.streamed_experts), s1.streamed_gib - s0.streamed_gib,
                    (long long) (s1.cpu_share_experts - s0.cpu_share_experts), s1.refill_ms - s0.refill_ms, s1.pinned_layers);
    }
    if (hash_state) {
        const EngineSnapshot snap = engine.snapshot();
        std::printf("state hash %016llx (%zu bytes), logits hash %016llx\n", (unsigned long long) fnv1a(snap.state.data(), snap.state.size()),
                    snap.state.size(), (unsigned long long) fnv1a(logits.data(), logits.size() * sizeof(float)));
    }
    if (hash) std::printf("hash prompt logits: %016llx\n", (unsigned long long) fnv1a(logits.data(), logits.size() * sizeof(float)));
    print_kv(engine, "after the prompt");
    for (std::int32_t n : followups) {
        const std::vector<std::int32_t> extra(prompt.begin(), prompt.begin() + std::min<std::size_t>(prompt.size(), std::size_t(n)));
        const std::int64_t at = engine.n_past();
        const auto tf = std::chrono::steady_clock::now();
        logits = engine.forward(extra);
        const double sec = seconds_since(tf);
        std::printf("follow-up: %zu tokens at depth %lld in %.3f s (%.1f tok/s)", extra.size(), (long long) at, sec, double(extra.size()) / sec);
        if (hash) std::printf("; hash logits %016llx", (unsigned long long) fnv1a(logits.data(), logits.size() * sizeof(float)));
        std::printf("\n");
    }
    if (!followups.empty()) print_kv(engine, "after the follow-ups");

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
        // the 8 tokens, the logits after them, and the logits each token was chosen from
        std::vector<std::vector<float>> seen[2];
        int run_no = 0;
        auto greedy = [&](std::vector<float> lg) {
            std::vector<std::int32_t> ids;
            for (int i = 0; i < 8; ++i) {
                seen[run_no].push_back(lg);
                const std::int32_t id = top_k(lg.data(), std::size_t(engine.n_vocab()), 1)[0].id;
                ids.push_back(id);
                lg = engine.forward({id});
            }
            seen[run_no].push_back(lg);
            ++run_no;
            return std::make_pair(ids, lg);
        };
        const auto first = greedy(logits);
        if (detour > 0) {
            std::vector<std::int32_t> extra;
            while (int(extra.size()) < detour) extra.insert(extra.end(), prompt.begin(), prompt.begin() + std::min<std::size_t>(prompt.size(), std::size_t(detour) - extra.size()));
            engine.forward(extra);
            std::printf("snapshot detour: %d more tokens fed before the restore\n", detour);
        }
        bool park_checks = true;  // with --test-park: a restore without the unpark must be refused
        if (test_park) {
            auto t_park = std::chrono::steady_clock::now();
            const EngineParked parked = engine.park(std::int64_t(prompt.size()));
            const double park_ms = 1e3 * seconds_since(t_park);
            // another sequence over every parked position: what an unrelated request does to the caches
            std::vector<std::int32_t> other(prompt.rbegin(), prompt.rend());
            for (std::size_t i = 0; other.size() < prompt.size() + 64; ++i) other.push_back(prompt[i % prompt.size()]);
            if (std::int64_t(other.size()) > opt.max_ctx) other.resize(std::size_t(opt.max_ctx));
            engine.reset();
            engine.forward(other);
            bool stale_rejected = false;
            try {
                engine.restore(snap);
            } catch (const std::exception &) {
                stale_rejected = true;  // the caches no longer hold the snapshot's tokens
            }
            t_park = std::chrono::steady_clock::now();
            engine.unpark(parked);
            const double unpark_ms = 1e3 * seconds_since(t_park);
            std::printf("park: %.2f GiB in %.1f ms, %zu other tokens fed, restore without unpark rejected: %s, unpark in %.1f ms\n",
                        parked.bytes() / 1073741824.0, park_ms, other.size(), stale_rejected ? "yes" : "NO", unpark_ms);
            park_checks = stale_rejected;
        }
        t_snap = std::chrono::steady_clock::now();
        engine.restore(snap);
        const double restore_ms = 1e3 * seconds_since(t_snap);
        const auto second = greedy(logits);
        // The second continuation is bitwise the first only if the expert cache did not change in between: with decode
        // adaptation (the default) it changes during the first continuation, and a detour's prompt refills and re-ranks it
        // (unless there is no cache), so some pairs then run on the other device: the same tokens, logits that differ in
        // the last bits. Bitwise: --no-decode-adapt, and without a detour or with --cache-mib 0.
        const bool strict = !opt.decode_adapt && ((detour == 0 && !test_park) || opt.expert_cache_mib == 0);
        const bool same_ids = first.first == second.first, same = same_ids && first.second == second.second;
        std::printf("snapshot %.1f MiB in %.1f ms, restore in %.1f ms; continuation after restore: tokens identical: %s, logits bitwise "
                    "identical: %s%s\n",
                    snap.state.size() / 1048576.0, snap_ms, restore_ms, same_ids ? "yes" : "NO", same ? "yes" : "no",
                    strict ? "" : opt.decode_adapt ? " (the expert cache adapts while decoding)" : " (the detour changed the expert cache)");
        if (!same) {
            // how far apart: the largest logit difference over the 9 rows, and the closest top-1/top-2 margin the
            // greedy choices had (a token flips only where the difference reaches the margin)
            double maxd = 0, min_margin = 1e30;
            int first_diff = -1;
            for (std::size_t r = 0; r < seen[0].size() && r < seen[1].size(); ++r) {
                const std::vector<float> & a = seen[0][r];
                const std::vector<float> & b = seen[1][r];
                for (std::size_t i = 0; i < a.size(); ++i) maxd = std::max(maxd, std::fabs(double(a[i]) - b[i]));
                if (r < 8) {
                    const auto t2 = top_k(a.data(), a.size(), 2);
                    min_margin = std::min(min_margin, double(t2[0].logit) - t2[1].logit);
                }
                if (first_diff < 0 && r < 8 && first.first[r] != second.first[r]) first_diff = int(r);
            }
            std::printf("snapshot continuations differ by at most %.3g in a logit; smallest top-1/top-2 margin of the 8 choices %.3g%s\n", maxd,
                        min_margin, first_diff >= 0 ? (", first different token at " + std::to_string(first_diff)).c_str() : "");
        }
        engine.restore(snap);
        if (!same_ids || (!same && strict) || !park_checks) return 1;
    }

    std::vector<std::int32_t> generated;
    std::vector<std::vector<Top>> tops;
    std::vector<double> step_times;
    const EngineStats after_prompt = engine.stats();
    const std::size_t V = std::size_t(engine.n_vocab());
    // The decode phase's expert statistics (the refill and re-ranking after a long prompt happen at the first decode
    // step, so they are counted here, not in the prompt's line).
    auto print_decode_experts = [&]() {
        const EngineStats & st = engine.stats();
        const std::int64_t hits = st.expert_hits - after_prompt.expert_hits, pairs = st.expert_pairs - after_prompt.expert_pairs;
        std::int64_t steps = 0;
        std::printf("decode experts: %.1f%% VRAM (%lld of %lld pairs); steps by tokens:", 100.0 * double(hits) / double(std::max<std::int64_t>(1, pairs)),
                    (long long) hits, (long long) pairs);
        for (int t = 1; t <= 4; ++t) {
            const std::int64_t n = st.graph_steps[t] - after_prompt.graph_steps[t];
            steps += n;
            std::printf(" %d:%lld", t, (long long) n);
        }
        steps = std::max<std::int64_t>(1, steps);
        std::printf("; per step: GPU wait %.2f ms, CPU experts %.2f ms, %.1f distinct CPU experts per layer; cache: %lld swaps (%lld in the "
                    "background, %.1f per step, chosen in %.3f ms; %.0f ms for the others), refill %.0f ms\n",
                    (st.gpu_wait_ms - after_prompt.gpu_wait_ms) / double(steps), (st.cpu_experts_ms - after_prompt.cpu_experts_ms) / double(steps),
                    double(st.cpu_expert_reads - after_prompt.cpu_expert_reads) / double(steps) / 48.0,
                    (long long) (st.cache_swaps - after_prompt.cache_swaps), (long long) (st.decode_swaps - after_prompt.decode_swaps),
                    double(st.decode_swaps - after_prompt.decode_swaps) / double(steps), (st.adapt_ms - after_prompt.adapt_ms) / double(steps),
                    st.cache_swap_ms - after_prompt.cache_swap_ms,
                    st.refill_ms - after_prompt.refill_ms);
    };
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
            if (hash) h_drafts = fnv1a(d.data(), d.size() * sizeof(std::int32_t), h_drafts);
            seq.insert(seq.end(), d.begin(), d.end());
            auto tv = std::chrono::steady_clock::now();
            const std::vector<float> lg = engine.forward(seq, true);
            verify_s += seconds_since(tv);
            if (hash) h_decode = fnv1a(lg.data(), lg.size() * sizeof(float), h_decode);  // 3 MB: only when asked (it is timed)
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
        const EngineStats & es = engine.stats();
        std::printf("per step: draft %.2f ms (catch-up %.2f, launches %.2f, GPU before the experts %.2f, MTP experts on the CPU %.2f, GPU after "
                    "them %.2f), verify %.2f ms (CPU experts %.2f ms; queueing swaps %.2f ms), other %.2f ms\n",
                    1e3 * draft_s / steps, (es.mtp_catchup_ms - after_prompt.mtp_catchup_ms) / steps,
                    (es.mtp_launch_ms - after_prompt.mtp_launch_ms) / steps, (es.mtp_wait_ms - after_prompt.mtp_wait_ms) / steps,
                    (es.mtp_cpu_ms - after_prompt.mtp_cpu_ms) / steps, (es.mtp_tail_ms - after_prompt.mtp_tail_ms) / steps, 1e3 * verify_s / steps,
                    (es.cpu_experts_ms - cpu0) / steps, (es.swap_issue_ms - after_prompt.swap_issue_ms) / steps, 1e3 * (total - draft_s - verify_s) / steps);
        print_decode_experts();
        if (hash)
            std::printf("hash verify logits: %016llx; hash drafts: %016llx\n", (unsigned long long) h_decode, (unsigned long long) h_drafts);
        print_kv(engine, "at the end");
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
        if (hash) h_decode = fnv1a(logits.data(), logits.size() * sizeof(float), h_decode);
    }
    std::printf("generated:");
    for (std::int32_t id : generated) std::printf(" %d", id);
    std::printf("\n");
    if (hash) std::printf("hash decode logits: %016llx\n", (unsigned long long) h_decode);
    print_kv(engine, "at the end");
    if (!step_times.empty()) {
        const double total = std::accumulate(step_times.begin(), step_times.end(), 0.0);
        const EngineStats & st = engine.stats();
        std::vector<double> sorted = step_times;
        std::sort(sorted.begin(), sorted.end());
        std::printf("step times: median %.2f ms, p90 %.2f ms, max %.2f ms (step %d)\n", 1e3 * sorted[sorted.size() / 2],
                    1e3 * sorted[sorted.size() * 9 / 10], 1e3 * sorted.back(),
                    int(std::max_element(step_times.begin(), step_times.end()) - step_times.begin()));
        const double cpu_ms = st.cpu_experts_ms - after_prompt.cpu_experts_ms, eng_ms = st.step_ms - after_prompt.step_ms;
        const std::int64_t hits = st.expert_hits - after_prompt.expert_hits, pairs = st.expert_pairs - after_prompt.expert_pairs;
        const std::int64_t host = st.expert_host_reads - after_prompt.expert_host_reads;
        std::printf("decode: %zu tokens, %.2f ms/token (%.2f tok/s); CPU experts %.1f%% of engine time; experts: %.1f%% VRAM, %.1f%% GPU from "
                    "host memory, %.1f%% CPU\n",
                    step_times.size(), 1e3 * total / double(step_times.size()), double(step_times.size()) / total, 100.0 * cpu_ms / eng_ms,
                    100.0 * double(hits) / double(std::max<std::int64_t>(1, pairs)), 100.0 * double(host) / double(std::max<std::int64_t>(1, pairs)),
                    100.0 * double(pairs - hits - host) / double(std::max<std::int64_t>(1, pairs)));
        print_decode_experts();
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
