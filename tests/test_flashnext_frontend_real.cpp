// The Qwen3.5 frontend over a real Flash-Next GGUF, checked against llama.cpp.
//
// NINFER_FLASHNEXT_GGUF names shard 1 of the model. NINFER_FLASHNEXT_ORACLE names a directory of
// <name>.ids files, comma-separated token ids from the llama.cpp server's /tokenize: every file
// must survive decode-then-encode unchanged, and chat.ids (when present) must equal the frontend's
// rendering of its own decoded user message with thinking disabled. Skips (77) when unset.
#include "flashnext/gguf.h"
#include "flashnext/shards.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "runtime/engine/flashnext_frontend.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

std::vector<int> read_ids(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    std::vector<int> ids;
    std::string item;
    while (std::getline(text, item, ',')) {
        if (item.find_first_not_of(" \r\n\t") != std::string::npos) {
            ids.push_back(std::stoi(item));
        }
    }
    return ids;
}

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    const char* model_path  = std::getenv("NINFER_FLASHNEXT_GGUF");
    const char* oracle_path = std::getenv("NINFER_FLASHNEXT_ORACLE");
    if (model_path == nullptr || oracle_path == nullptr) {
        std::cout << "skipped: NINFER_FLASHNEXT_GGUF and NINFER_FLASHNEXT_ORACLE are unset\n";
        return 77;
    }
    const flashnext::GgufModel model(flashnext::gguf_shard_paths(model_path));
    const runtime::FlashNextFrontendFiles files = runtime::flashnext_frontend_files(model);
    const auto rows = static_cast<std::uint32_t>(model.tensor("token_embd.weight").shape[1]);
    const models::qwen3_5::FrontendResources resources =
        runtime::flashnext_frontend_resources(files, rows);
    const auto& tokenizer = *resources.tokenizer;

    int failures = 0;
    failures += check(resources.public_token_count == 248077 && rows == 248320,
                      "unexpected public token domain");
    const std::vector<int> stops = tokenizer.default_stop_token_ids();
    failures += check(stops == std::vector<int>{248046, 248044},
                      "end-of-generation tokens are not <|im_end|>, <|endoftext|>");
    failures += check(tokenizer.is_special_token(248045) && !tokenizer.is_special_token(248058) &&
                          !tokenizer.is_special_token(248068),
                      "control and plain added tokens are not separated");

    for (const auto& entry : std::filesystem::directory_iterator(oracle_path)) {
        if (entry.path().extension() != ".ids") { continue; }
        const std::vector<int> ids  = read_ids(entry.path());
        const std::string text      = tokenizer.decode(ids);
        const std::vector<int> back = tokenizer.encode(text);
        failures += check(back == ids, entry.path().filename().string() +
                                           ": decode then encode differs from llama.cpp");
        std::cout << entry.path().filename().string() << ": " << ids.size() << " tokens "
                  << (back == ids ? "match" : "DIFFER") << '\n';
    }

    const std::filesystem::path chat = std::filesystem::path(oracle_path) / "chat.ids";
    if (std::filesystem::exists(chat)) {
        // <|im_start|>user\n TEXT <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
        const std::vector<int> ids = read_ids(chat);
        const std::vector<int> user(ids.begin() + 3, ids.end() - 9);
        const auto frontend = models::qwen3_5::make_frontend(
            resources, {.architecture   = models::Architecture::Qwen3_5Moe,
                        .vision_enabled = false,
                        .max_context    = 4096});
        PromptInput input;
        ChatMessage message;
        message.role = ChatRole::User;
        message.parts.push_back(
            MessagePart{.kind = MessagePartKind::Text, .text = tokenizer.decode(user)});
        input.messages.push_back(std::move(message));
        input.options.enable_thinking = false;
        const auto prompt             = frontend.prepare(std::move(input));
        const auto& rendered = models::qwen3_5::PreparedPromptAccess::view(prompt).token_ids;
        failures += check(std::vector<int>(rendered.begin(), rendered.end()) == ids,
                          "chat template rendering differs from llama.cpp");
        std::cout << "chat.ids: rendered " << rendered.size() << " tokens, expected " << ids.size()
                  << '\n';
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
