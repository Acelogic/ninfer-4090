// Read-only GGUF v3 access for Qwen3.8-Flash-Next: metadata, tensor directory, and memory-mapped data.
//
// A split model (Unsloth ships three shards) is opened as one: the first shard carries the metadata,
// every shard carries part of the tensor directory, and each tensor's bytes are mapped in place.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::flashnext {

// ggml tensor type ids used by the Flash-Next GGUFs.
enum class GgufType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q6_K = 14,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ4_XS = 23,
    BF16 = 30,
};

const char * type_name(GgufType type);
// Elements per block and bytes per block; throws for a type this runtime does not read.
std::int64_t block_elems(GgufType type);
std::size_t block_bytes(GgufType type);
std::size_t row_bytes(GgufType type, std::int64_t elems);

struct GgufTensor {
    std::string name;
    GgufType type{};
    std::vector<std::int64_t> shape;  // ne[0] is the contiguous (row) dimension, as in ggml
    const std::uint8_t * data = nullptr;
    std::size_t bytes = 0;
    std::int64_t elements() const;
};

using GgufValue = std::variant<std::int64_t, std::uint64_t, double, bool, std::string,
                               std::vector<std::int64_t>, std::vector<std::uint64_t>, std::vector<double>,
                               std::vector<std::string>>;

class MappedFile;

class GgufModel {
public:
    // Opens every shard; `paths` may list them in any order.
    explicit GgufModel(const std::vector<std::string> & paths);
    ~GgufModel();
    GgufModel(const GgufModel &) = delete;
    GgufModel & operator=(const GgufModel &) = delete;

    bool has(const std::string & key) const { return kv_.count(key) != 0; }
    std::int64_t get_int(const std::string & key) const;
    double get_float(const std::string & key) const;
    std::string get_string(const std::string & key) const;
    std::vector<std::int64_t> get_ints(const std::string & key) const;
    std::vector<std::uint64_t> get_uints(const std::string & key) const;
    std::vector<std::string> get_strings(const std::string & key) const;

    const GgufTensor * find(const std::string & name) const;
    const GgufTensor & tensor(const std::string & name) const;  // throws when missing
    const std::map<std::string, GgufTensor> & tensors() const { return tensors_; }

private:
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::map<std::string, GgufValue> kv_;
    std::map<std::string, GgufTensor> tensors_;
};

}  // namespace ninfer::flashnext
