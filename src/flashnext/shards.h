// Shard discovery for split GGUF models.
#pragma once
#include <string>
#include <vector>

namespace ninfer::flashnext {

// Every shard of the split GGUF that `path` belongs to, in order, when its file name follows the
// <stem>-0000i-of-0000n.gguf convention (any shard may be given); otherwise just `path`. Throws when
// a listed shard is missing.
std::vector<std::string> gguf_shard_paths(const std::string & path);

}  // namespace ninfer::flashnext
