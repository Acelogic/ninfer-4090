#include "flashnext/shards.h"

#include <cstdio>
#include <filesystem>
#include <regex>
#include <stdexcept>

namespace ninfer::flashnext {

std::vector<std::string> gguf_shard_paths(const std::string & path) {
    namespace fs = std::filesystem;
    static const std::regex pattern(R"((.*)-(\d{5})-of-(\d{5})\.gguf)");
    const fs::path first(path);
    const std::string name = first.filename().string();
    std::smatch match;
    if (!std::regex_match(name, match, pattern)) return {path};
    const int count = std::stoi(match[3].str());
    if (count < 1) throw std::invalid_argument(path + ": invalid GGUF shard count");
    std::vector<std::string> shards;
    for (int i = 1; i <= count; ++i) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, count);
        const fs::path shard = first.parent_path() / (match[1].str() + suffix);
        if (!fs::is_regular_file(shard)) throw std::invalid_argument("GGUF shard is missing: " + shard.string());
        shards.push_back(shard.string());
    }
    return shards;
}

}  // namespace ninfer::flashnext
