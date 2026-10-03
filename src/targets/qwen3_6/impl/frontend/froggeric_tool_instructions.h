#pragma once
#include <string_view>
namespace ninfer::targets::qwen3_6::frontend_internal {
inline constexpr std::string_view kFroggericThinkingToolInstructions = R"FG(

If you choose to call a function ONLY reply in the following format with NO suffix:

<think>
Brief explanation of tool call
</think>
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.
- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.
- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.
- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or indentation before them.
- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.
- If you have all necessary data, provide your final answer directly to the user without any tool call.
</IMPORTANT>)FG";
inline constexpr std::string_view kFroggericNonThinkingToolInstructions = R"FG(

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.
- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO conversational text before it.
- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or indentation before them.
- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.
- If you have all necessary data, provide your final answer directly to the user without any tool call.
</IMPORTANT>)FG";
}
