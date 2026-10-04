// Checks the Flash-Next GGUF reader and dequantizers: prints the model's metadata, compares our
// dequantization with ggml's on sample rows of every tensor format (bit-exact), and checks that the
// IQ3_S -> Q4L repack round-trips exactly. ggml is linked here only as a test oracle.
//
// Usage: check_formats <shard> [<shard> ...]
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include "flashnext/gguf.h"
#include "flashnext/quants.h"
#include "ggml.h"

using namespace ninfer::flashnext;

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> paths(argv + 1, argv + argc);
    GgufModel m(paths);
    const std::string arch = m.get_string("general.architecture");
    printf("architecture %s, %zu tensors\n", arch.c_str(), m.tensors().size());
    for (const char * k : {"block_count", "embedding_length", "expert_count", "expert_used_count", "attention.head_count",
                           "attention.head_count_kv", "attention.key_length", "ssm.state_size", "hyper_connection.count",
                           "attention.indexer.top_k", "ple.ngram_size", "embedding_length_per_layer_input"}) {
        printf("  %s.%s = %lld\n", arch.c_str(), k, (long long) m.get_int(arch + "." + k));
    }
    printf("  rope.freq_base = %g, rms eps = %g\n", m.get_float(arch + ".rope.freq_base"), m.get_float(arch + ".attention.layer_norm_rms_epsilon"));

    // One sample tensor per type; compare up to 64 random rows against ggml.
    std::map<GgufType, const GgufTensor *> sample;
    for (const auto & [name, t] : m.tensors()) sample.emplace(t.type, &t);
    std::mt19937 rng(11);
    int failures = 0;
    for (const auto & [type, t] : sample) {
        if (type == GgufType::F32) continue;  // ggml has no to_float for f32; ours is a plain copy
        const std::int64_t row = t->shape[0], rows = t->elements() / row;
        const std::size_t rb = row_bytes(type, row);
        std::vector<float> ours(std::size_t(row), 0.f), theirs(std::size_t(row), 0.f);
        int bad = 0;
        for (int s = 0; s < 64; ++s) {
            const std::int64_t r = std::int64_t(rng() % std::uint64_t(rows));
            const std::uint8_t * src = t->data + std::size_t(r) * rb;
            dequantize_row(type, src, ours.data(), row);
            ggml_get_type_traits(ggml_type(type))->to_float(src, theirs.data(), row);
            if (std::memcmp(ours.data(), theirs.data(), ours.size() * 4) != 0) ++bad;
        }
        printf("%-8s %-40s %s\n", type_name(type), t->name.c_str(), bad ? "MISMATCH" : "bit-exact vs ggml");
        failures += bad != 0;
    }

    // Q4L round trip on a real expert tensor.
    const GgufTensor & g = m.tensor("blk.0.ffn_gate_exps.weight");
    const std::int64_t nblocks = std::int64_t(g.bytes / sizeof(BlockIQ3_S));
    const std::int64_t check = std::min<std::int64_t>(nblocks, 4096);
    std::vector<BlockQ4L> q4l(static_cast<std::size_t>(check));
    repack_iq3s_to_q4l(reinterpret_cast<const BlockIQ3_S *>(g.data), q4l.data(), check);
    std::vector<float> a(256), b(256);
    int q4l_bad = 0;
    for (std::int64_t i = 0; i < check; ++i) {
        dequantize_row(GgufType::IQ3_S, g.data + std::size_t(i) * sizeof(BlockIQ3_S), a.data(), 256);
        dequantize_row_q4l(&q4l[std::size_t(i)], b.data(), 256);
        if (std::memcmp(a.data(), b.data(), 256 * 4) != 0) ++q4l_bad;
    }
    printf("q4l repack: %lld blocks, %d differ from iq3_s\n", (long long) check, q4l_bad);
    failures += q4l_bad != 0;
    printf(failures ? "FAILED\n" : "ALL OK\n");
    return failures ? 1 : 0;
}
