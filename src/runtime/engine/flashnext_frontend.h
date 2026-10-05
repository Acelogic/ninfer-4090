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
    // With a vision encoder: the image (and video) preprocessor configurations of Qwen3-VL's pixel pipeline (patch
    // 16, merge 2, two frames, mean and std 0.5, bicubic resize to multiples of 32 within the pixel budget).
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
};

// The image pixel budget of a Flash-Next server, llama.cpp's for Qwen3-VL (mtmd's qwen3vl_merger: 8 to 4,096 tokens
// of 32 x 32 pixels): smaller images are scaled up, larger ones down, keeping the aspect ratio.
inline constexpr std::uint64_t kFlashNextImageMinPixels = 8ULL * 32ULL * 32ULL;
inline constexpr std::uint64_t kFlashNextImageMaxPixels = 4096ULL * 32ULL * 32ULL;

[[nodiscard]] FlashNextFrontendFiles flashnext_frontend_files(const flashnext::GgufModel& model,
                                                              bool vision = false);

// Parses the tokenizer and validates its public token domain against the embedding rows. The
// returned views borrow `files`.
[[nodiscard]] models::qwen3_5::FrontendResources
flashnext_frontend_resources(const FlashNextFrontendFiles& files, std::uint32_t embedding_rows);

} // namespace ninfer::runtime
