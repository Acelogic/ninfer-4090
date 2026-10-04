#pragma once

#include "models/qwen3_5/frontend/resources.h"

#include <cstdint>
#include <string>

namespace ninfer::flashnext {
class GgufModel;
}

namespace ninfer::runtime {

// The Qwen3.5 frontend files of a Flash-Next GGUF, generated from its tokenizer metadata: the
// byte-level BPE vocabulary and merges, the control (special) and user-defined (plain) added
// tokens, the end-of-generation tokens, and the embedded chat template. Ids past the public
// vocabulary (padding rows of the embedding) are left out, as the Hugging Face tokenizer leaves
// them out.
struct FlashNextFrontendFiles {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string generation_config_json;
    std::string chat_template;
};

[[nodiscard]] FlashNextFrontendFiles flashnext_frontend_files(const flashnext::GgufModel& model);

// Parses the tokenizer and validates its public token domain against the embedding rows. The
// returned views borrow `files`.
[[nodiscard]] models::qwen3_5::FrontendResources
flashnext_frontend_resources(const FlashNextFrontendFiles& files, std::uint32_t embedding_rows);

} // namespace ninfer::runtime
