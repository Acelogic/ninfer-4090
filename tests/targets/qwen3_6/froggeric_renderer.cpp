#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using Json = nlohmann::ordered_json;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    const std::string source{std::istreambuf_iterator<char>(file), {}};
    const auto renderer = fi::CompiledChatTemplate::resolve(source);
    for (std::string line; std::getline(std::cin, line);) {
        try {
            const auto input = Json::parse(line);
            std::vector<fi::ChatMessage> messages;
            for (const auto& m : input.at("messages")) {
                fi::ChatMessage msg;
                msg.role = m.at("role");
                auto content = m.value("content", Json());
                if (content.is_string()) msg.parts.push_back(fi::ChatPart::text_part(content.get<std::string>()));
                else if (content.is_array()) for (const auto& part : content) {
                    if (part.is_string()) msg.parts.push_back(fi::ChatPart::text_part(part.get<std::string>()));
                    else if (part.value("type", "") == "image") msg.parts.push_back(fi::ChatPart::image({}));
                    else if (part.value("type", "") == "video") msg.parts.push_back(fi::ChatPart::video({}));
                    else msg.parts.push_back(fi::ChatPart::text_part(part.at("text").get<std::string>()));
                }
                if (m.contains("reasoning_content") && !m["reasoning_content"].is_null()) msg.reasoning_content = m["reasoning_content"].get<std::string>();
                for (const auto& call : m.value("tool_calls", Json::array())) {
                    const auto fn = call.contains("function") ? call["function"] : call;
                    msg.tool_calls.push_back({.name = fn.at("name"), .arguments_json = fn.value("arguments", Json::object()).dump()});
                }
                messages.push_back(std::move(msg));
            }
            fi::ChatRenderOptions opts;
            const auto kw = input.value("options", Json::object());
            opts.add_generation_prompt = kw.value("add_generation_prompt", true);
            opts.enable_thinking = kw.value("enable_thinking", true);
            opts.add_vision_id = kw.value("add_vision_id", false);
            if (kw.contains("preserve_thinking")) opts.preserve_thinking = kw.at("preserve_thinking").get<bool>();
            if (kw.contains("reasoning_effort")) {
                const std::string effort = kw.at("reasoning_effort");
                opts.reasoning_effort = effort == "low" ? ninfer::ReasoningEffort::Low : effort == "medium" ? ninfer::ReasoningEffort::Medium : ninfer::ReasoningEffort::XHigh;
            }
            for (const auto& t : input.value("tools", Json::array())) opts.tool_jsons.push_back(t.dump());
            const auto rendered = renderer.render(messages, opts);
            std::cout << Json{{"text",rendered.text},{"starts_in_reasoning",rendered.starts_in_reasoning},
                {"rewrite",rendered.turn_rewrite_byte_offset ? Json(*rendered.turn_rewrite_byte_offset) : Json()}}.dump() << '\n';
        } catch (const std::exception& e) {
            std::cout << Json{{"error",e.what()}}.dump() << '\n';
        }
    }
}
