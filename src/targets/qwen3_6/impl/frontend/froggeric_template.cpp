// Native port of froggeric/Qwen-Fixed-Chat-Templates, qwen3.8-froggeric-v22.5.
// Upstream source and Apache-2.0 attribution: third_party/froggeric/.
// Implements the NInfer typed frontend contract: XML calls, parsed object arguments,
// explicit reasoning_content, and unlimited tool text. No Jinja subprocess at inference.
#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/froggeric_tool_instructions.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {
using Json = nlohmann::ordered_json;
constexpr std::array<std::string_view, 10> kTags = {
    "<|think_off|>", "<|think_on|>", "<|think_xhigh|>", "<|think_high|>",
    "<|think_ultracode|>", "<|think_extreme|>", "<|think_max|>",
    "<|think_medium|>", "<|think_low|>", "<|think_minimal|>"};
bool contains(std::string_view s, std::string_view part) { return s.find(part) != s.npos; }
std::string trim(std::string_view s) {
    // Python str.strip whitespace, including Unicode separators and C0 U+001C..1F.
    constexpr std::array<std::string_view, 18> unicode = {
        "\xc2\x85", "\xc2\xa0", "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
        "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86",
        "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
        "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f"};
    auto ascii = [](unsigned char c) { return c == 32 || (c >= 9 && c <= 13) || (c >= 28 && c <= 31); };
    for (;;) {
        if (s.empty()) return {};
        if (ascii(s.front())) { s.remove_prefix(1); continue; }
        bool changed = false;
        for (auto w : unicode) if (s.starts_with(w)) { s.remove_prefix(w.size()); changed = true; break; }
        if (changed) continue;
        if (s.starts_with("\xe3\x80\x80")) { s.remove_prefix(3); continue; }
        break;
    }
    for (;;) {
        if (s.empty()) return {};
        if (ascii(s.back())) { s.remove_suffix(1); continue; }
        bool changed = false;
        for (auto w : unicode) if (s.ends_with(w)) { s.remove_suffix(w.size()); changed = true; break; }
        if (changed) continue;
        if (s.ends_with("\xe3\x80\x80")) { s.remove_suffix(3); continue; }
        break;
    }
    return std::string(s);
}
std::string strip_tags(std::string s) {
    for (auto tag : kTags) {
        if (tag.empty() || !contains(s, tag)) continue;
        std::size_t at;
        while ((at = s.find(tag)) != s.npos) s.erase(at, tag.size());
        s = trim(s);
    }
    return s;
}
bool system_role(std::string_view role) { return role == "system" || role == "developer"; }
void scan_tags(std::string_view s, bool& thinking, ReasoningEffort& effort) {
    if (contains(s, "<|think_off|>")) thinking = false;
    else if (contains(s, "<|think_on|>")) thinking = true;
    else if (contains(s, kTags[2]) || contains(s, kTags[3]) || contains(s, kTags[4]) ||
             contains(s, kTags[5]) || contains(s, kTags[6])) {
        thinking = true; effort = ReasoningEffort::XHigh;
    } else if (contains(s, kTags[8]) || contains(s, kTags[9])) {
        thinking = true; effort = ReasoningEffort::Low;
    } else if (contains(s, kTags[7])) {
        thinking = true; effort = ReasoningEffort::Medium;
    }
}
std::string tojson(const Json& value) {
    if (value.is_array() || value.is_object()) {
        const bool object = value.is_object();
        std::string s = object ? "{" : "[";
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) s += ", ";
            first = false;
            if (object) s += Json(it.key()).dump() + ": ";
            s += tojson(it.value());
        }
        return s + (object ? "}" : "]");
    }
    return value.dump();
}
std::string lstrip_nl(std::string s) {
    const auto at = s.find_first_not_of('\n');
    return at == s.npos ? "" : s.substr(at);
}
std::string rstrip_nl(std::string s) {
    const auto at = s.find_last_not_of('\n');
    return at == s.npos ? "" : s.substr(0, at + 1);
}
std::pair<std::string, std::string> think_parts(std::string content, std::string explicit_reasoning) {
    std::string_view end;
    if (!explicit_reasoning.empty()) {
        if (content.starts_with("<think>") && contains(content, "</think>")) end = "</think>";
        else if (content.starts_with("<thinking>") && contains(content, "</thinking>")) end = "</thinking>";
        else if (content.starts_with("</think>")) end = "</think>";
        else if (content.starts_with("</thinking>")) end = "</thinking>";
        if (!end.empty()) content = lstrip_nl(content.substr(content.rfind(end) + end.size()));
        return {trim(explicit_reasoning), content};
    }
    if (content.starts_with("</think>")) end = "</think>";
    else if (content.starts_with("</thinking>")) end = "</thinking>";
    else if (contains(content, "\n</think>")) end = "\n</think>";
    else if (contains(content, "\n</thinking>")) end = "\n</thinking>";
    else if (contains(content, "\n</ think>")) end = "\n</ think>";
    else if (contains(content, "\n</think >")) end = "\n</think >";
    else if (content.starts_with("<think>") && contains(content, "</think>")) end = "</think>";
    else if (content.starts_with("<thinking>") && contains(content, "</thinking>")) end = "</thinking>";
    std::string reasoning;
    if (!end.empty()) {
        auto start = contains(end, "thinking") ? std::string_view("<thinking>") : std::string_view("<think>");
        reasoning = rstrip_nl(content.substr(0, content.find(end)));
        if (contains(reasoning, start)) reasoning = lstrip_nl(reasoning.substr(reasoning.rfind(start) + start.size()));
        content = lstrip_nl(content.substr(content.rfind(end) + end.size()));
    }
    return {trim(reasoning), content};
}
// UTF-8 character boundaries match Jinja's character-based head/length tests.
std::size_t utf8_length(std::string_view s) {
    return std::count_if(s.begin(), s.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
}
std::string_view head_chars(std::string_view s, std::size_t limit) {
    std::size_t chars = 0, i = 0;
    for (; i < s.size(); ++i) if ((static_cast<unsigned char>(s[i]) & 0xc0) != 0x80 && chars++ == limit) break;
    return s.substr(0, i);
}
bool tool_error(std::string content) {
    std::transform(content.begin(), content.end(), content.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto head = head_chars(content, 120);
    auto any = [](std::string_view s, std::initializer_list<std::string_view> needles) {
        for (auto n : needles) if (contains(s, n)) return true;
        return false;
    };
    const bool code = any(content, {"throw new ", "throw error", "console.error", "logger.error", "logging.error"}) ||
                      any(head, {"import ", "def ", "function "});
    const bool zero = any(head, {"exit code: 0", "process exited with code 0"});
    const bool error_ok = any(head, {"\"error\": null", "\"error\":null", "\"error\": false", "\"error\":false", "\"error\": \"\"", "\"error\":\"\""});
    const bool strong = (contains(head, "\"error\":") && !error_ok) ||
        any(head, {"\"status\": \"error\"", "\"status\":\"error\"", "traceback (most recent call last):", "command not found", "invalid syntax", "fatal:"}) ||
        (any(head, {"exit code: ", "process exited with code"}) && !zero) || head.starts_with("exception:") || head.starts_with("failed to ");
    const bool weak = any(head, {"error:", "err!"});
    const bool suppressed = any(head, {"$ ", "took "}) || utf8_length(content) >= 600;
    return !code && (strong || (weak && !suppressed));
}
}

RenderedChat render_froggeric225(const std::vector<ChatMessage>& messages, ChatRenderOptions options) {
    if (messages.empty()) throw std::invalid_argument("chat messages must not be empty");
    bool thinking = options.enable_thinking;
    auto effort = options.reasoning_effort.value_or(ReasoningEffort::Medium);
    for (const auto& m : messages) if (system_role(m.role) || m.role == "user") {
        for (const auto& part : m.parts) if (part.kind == ChatPartKind::Text) scan_tags(part.text, thinking, effort);
    }
    std::string instructions;
    if (thinking && effort == ReasoningEffort::Low)
        instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    else if (thinking && effort == ReasoningEffort::XHigh)
        instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    std::size_t head = 0;
    std::string system;
    while (head < messages.size() && system_role(messages[head].role)) {
        const auto& m = messages[head++];
        if (m.has_media()) throw std::invalid_argument("system message cannot contain images or videos");
        auto part = strip_tags(trim(m.rendered_content()));
        if (!system.empty() && !part.empty()) system += "\n\n";
        system += part;
    }
    std::string out;
    if (!options.tool_jsons.empty()) {
        out = "<|im_start|>system\n";
        if (!instructions.empty()) out += instructions + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& tool : options.tool_jsons) out += "\n" + tojson(Json::parse(tool));
        out += "\n</tools>";
        out += thinking ? kFroggericThinkingToolInstructions : kFroggericNonThinkingToolInstructions;
        if (!system.empty()) out += "\n\n" + system;
        out += "<|im_end|>\n";
    } else if (!system.empty() || !instructions.empty()) {
        out = "<|im_start|>system\n" + instructions;
        if (!instructions.empty() && !system.empty()) out += "\n\n";
        out += system + "<|im_end|>\n";
    }
    const long last = static_cast<long>(messages.size()) - 1;
    long last_query = last - static_cast<long>(head) > 50 ? last : static_cast<long>(head);
    for (long i = last; i >= static_cast<long>(head); --i) if (messages[i].role == "user") {
        const auto s = trim(messages[i].rendered_content());
        if (!(s.starts_with("<tool_response>") && s.ends_with("</tool_response>"))) { last_query = i; break; }
    }
    std::optional<std::size_t> rewrite;
    int images = 0, videos = 0, failures = 0;
    for (std::size_t i = head; i < messages.size(); ++i) {
        const auto& m = messages[i];
        const bool sys = system_role(m.role);
        if (sys && m.has_media()) throw std::invalid_argument("system message cannot contain images or videos");
        auto content = trim(m.rendered_content(options.add_vision_id, &images, &videos));
        if (sys || m.role == "user") content = strip_tags(content);
        if (sys) out += "<|im_start|>system\n" + content + "<|im_end|>\n";
        else if (m.role == "user") { failures = 0; out += "<|im_start|>user\n" + content + "<|im_end|>\n"; }
        else if (m.role == "assistant") {
            auto [reasoning, body] = think_parts(content, m.reasoning_content);
            out += "<|im_start|>assistant\n";
            if (!rewrite && static_cast<long>(i) > last_query) rewrite = out.size();
            if (options.preserve_thinking.value_or(true) || static_cast<long>(i) > last_query)
                out += "<think>\n" + reasoning + "\n</think>\n\n";
            out += body;
            for (std::size_t j = 0; j < m.tool_calls.size(); ++j) {
                const auto& call = m.tool_calls[j];
                if (j > 0) out += "\n";
                else if (!trim(body).empty()) out += "\n\n";
                out += "<tool_call>\n<function=" + call.name + ">\n";
                const auto args = call.arguments_json.empty() ? Json::object() : Json::parse(call.arguments_json);
                if (!args.is_object()) throw std::invalid_argument("tool call arguments must be a JSON object");
                for (auto it = args.begin(); it != args.end(); ++it)
                    out += "<parameter=" + it.key() + ">\n" + (it.value().is_string() ? it.value().get<std::string>() : tojson(it.value())) + "\n</parameter>\n";
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            failures = tool_error(content) ? failures + 1 : 0;
            if (i == head || messages[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content;
            if (failures >= 2) out += "\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: " + std::to_string(failures) + " consecutive tool errors detected. Your previous approach is incorrect. You MUST use a fundamentally different approach or corrected arguments.";
            else if (failures == 1) out += "\n\n\xe2\x9a\xa0\xef\xb8\x8f SYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and retry with completely corrected arguments.";
            out += "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") out += "<|im_end|>\n";
        } else out += "<|im_start|>user\n[" + m.role + "]: " + content + "<|im_end|>\n";
    }
    if (options.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!rewrite) rewrite = out.size();
        out += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    }
    return {.text = std::move(out), .turn_rewrite_byte_offset = rewrite,
            .enable_thinking = thinking, .starts_in_reasoning = options.add_generation_prompt && thinking};
}
}
