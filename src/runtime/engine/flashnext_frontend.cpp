#include "runtime/engine/flashnext_frontend.h"

#include "flashnext/gguf.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {
namespace {

using Json = nlohmann::json;

// Token types of the GGUF tokenizer metadata (gguf-py TokenType).
enum GgufTokenType : std::int64_t {
    kNormal      = 1,
    kUnknown     = 2,
    kControl     = 3,
    kUserDefined = 4,
    kUnused      = 5,
    kByte        = 6,
};

std::int64_t optional_int(const flashnext::GgufModel& model, const char* key) {
    return model.has(key) ? model.get_int(key) : -1;
}

} // namespace

FlashNextFrontendFiles flashnext_frontend_files(const flashnext::GgufModel& model, bool vision) {
    if (model.get_string("tokenizer.ggml.model") != "gpt2") {
        throw std::invalid_argument("Flash-Next GGUF tokenizer is not byte-level BPE");
    }
    // The Qwen3.5 frontend implements the qwen35 pre-tokenizer (NFC, the Qwen3.5 split pattern and
    // byte-level mapping) when tokenizer.json leaves the pipeline implicit.
    const std::string pre =
        model.has("tokenizer.ggml.pre") ? model.get_string("tokenizer.ggml.pre") : std::string();
    if (pre != "qwen35") {
        throw std::invalid_argument("Flash-Next GGUF pre-tokenizer '" + pre +
                                    "' is not the Qwen3.5 tokenizer");
    }
    const std::vector<std::string> tokens = model.get_strings("tokenizer.ggml.tokens");
    const std::vector<std::int64_t> types = model.get_ints("tokenizer.ggml.token_type");
    if (tokens.empty() || types.size() != tokens.size()) {
        throw std::invalid_argument("Flash-Next GGUF tokenizer tokens and types differ in length");
    }

    Json vocab = Json::object();
    Json added = Json::array();
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        switch (types[id]) {
        case kNormal:
        case kByte:
            vocab[tokens[id]] = id;
            break;
        case kControl:
        case kUserDefined:
            added.push_back({{"id", id},
                             {"content", tokens[id]},
                             {"single_word", false},
                             {"lstrip", false},
                             {"rstrip", false},
                             {"normalized", false},
                             {"special", types[id] == kControl}});
            break;
        case kUnknown:
        case kUnused:
            break;
        default:
            throw std::invalid_argument("Flash-Next GGUF tokenizer has an unknown token type");
        }
    }
    Json tokenizer = {
        {"version", "1.0"},
        {"added_tokens", std::move(added)},
        {"model",
         {{"type", "BPE"},
          {"vocab", std::move(vocab)},
          {"merges", model.get_strings("tokenizer.ggml.merges")}}},
    };

    const auto token_text = [&](std::int64_t id) -> Json {
        if (id < 0) { return nullptr; }
        if (static_cast<std::size_t>(id) >= tokens.size()) {
            throw std::invalid_argument("Flash-Next GGUF special token id is out of range");
        }
        return tokens[static_cast<std::size_t>(id)];
    };
    const std::int64_t eos = model.get_int("tokenizer.ggml.eos_token_id");
    Json config            = {
        {"add_bos_token", model.has("tokenizer.ggml.add_bos_token") &&
                              model.get_int("tokenizer.ggml.add_bos_token") != 0},
        {"add_prefix_space", false},
        {"added_tokens_decoder", Json::object()},
        {"bos_token", nullptr},
        {"eos_token", token_text(eos)},
        {"pad_token", token_text(optional_int(model, "tokenizer.ggml.padding_token_id"))},
    };

    // Generation ends at the GGUF's EOS and end-of-turn tokens and at Qwen's end-of-turn and
    // end-of-text control tokens, the set llama.cpp also treats as end of generation.
    std::vector<std::int64_t> stops;
    const auto add_stop = [&](std::int64_t id) {
        if (id >= 0 && std::find(stops.begin(), stops.end(), id) == stops.end()) {
            stops.push_back(id);
        }
    };
    add_stop(eos);
    add_stop(optional_int(model, "tokenizer.ggml.eot_token_id"));
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        if (types[id] == kControl &&
            (tokens[id] == "<|im_end|>" || tokens[id] == "<|endoftext|>")) {
            add_stop(static_cast<std::int64_t>(id));
        }
    }

    FlashNextFrontendFiles files;
    files.tokenizer_json         = tokenizer.dump();
    files.tokenizer_config_json  = config.dump();
    files.generation_config_json = Json{{"eos_token_id", stops}}.dump();
    files.chat_template          = model.get_string("tokenizer.chat_template");
    if (vision) {
        const Json pipeline = {
            {"do_resize", true},
            {"do_rescale", true},
            {"do_normalize", true},
            {"do_convert_rgb", true},
            {"resample", 3},
            {"rescale_factor", 1.0 / 255.0},
            {"image_mean", {0.5, 0.5, 0.5}},
            {"image_std", {0.5, 0.5, 0.5}},
            {"patch_size", 16},
            {"temporal_patch_size", 2},
            {"merge_size", 2},
        };
        Json image   = pipeline;
        image["size"] = {{"shortest_edge", kFlashNextImageMinPixels}, {"longest_edge", kFlashNextImageMaxPixels}};
        Json video   = pipeline;
        video["size"] = {{"shortest_edge", 128ULL * 32ULL * 32ULL}, {"longest_edge", 4ULL * 1024ULL * 1024ULL}};
        video["fps"]        = 2.0;
        video["min_frames"] = 4;
        video["max_frames"] = 768;
        files.preprocessor_config_json       = image.dump();
        files.video_preprocessor_config_json = video.dump();
    }
    return files;
}

models::qwen3_5::FrontendResources flashnext_frontend_resources(const FlashNextFrontendFiles& files,
                                                                std::uint32_t embedding_rows) {
    models::qwen3_5::FrontendResources resources;
    resources.tokenizer_json         = files.tokenizer_json;
    resources.tokenizer_config_json  = files.tokenizer_config_json;
    resources.chat_template_jinja    = files.chat_template;
    resources.generation_config_json = files.generation_config_json;
    resources.preprocessor_config_json       = files.preprocessor_config_json;
    resources.video_preprocessor_config_json = files.video_preprocessor_config_json;
    resources.tokenizer              = std::make_shared<const models::qwen3_5::frontend::Tokenizer>(
        models::qwen3_5::frontend::TokenizerResources{
            files.tokenizer_json, files.tokenizer_config_json, files.generation_config_json});
    const std::size_t count = resources.tokenizer->vocab_size();
    if (count == 0 || count > embedding_rows ||
        !resources.tokenizer->has_exact_token_domain(count)) {
        throw std::invalid_argument("Flash-Next tokenizer must expose a contiguous public domain "
                                    "within the embedding rows");
    }
    resources.public_token_count = static_cast<std::uint32_t>(count);
    for (const int token : resources.tokenizer->default_stop_token_ids()) {
        if (!resources.tokenizer->is_valid_token(token)) {
            throw std::invalid_argument("Flash-Next stop token is outside the public token domain");
        }
    }
    return resources;
}

} // namespace ninfer::runtime
