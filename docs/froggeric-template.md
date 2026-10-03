# Froggeric v22.5 native template

`ninfer-serve --chat-template PATH` selects a hash-registered template at startup.
Omitting the flag preserves the artifact's original template. The weights and
embedded tokenizer/template resources are never rewritten. An unreadable, empty,
oversized, or unknown template is rejected rather than silently ignored.

This build registers the Apache-2.0 Qwen Fixed Chat Templates revision
`855bffc49448e299789730ff92c9b8d834d6cc14`, SHA-256
`e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`.
The original Jinja, upstream README, attribution, and license are in
`third_party/froggeric`. `froggeric_template.cpp` implements the template for
NInfer's typed frontend. There is no Python process or Jinja interpreter in the
inference path.

The serving contract uses XML tool calls, JSON-object arguments after protocol
normalization, `reasoning_content`, low/medium/xhigh reasoning, preserve-thinking,
vision IDs, and unlimited tool text. Froggeric's inline thinking tags are also
implemented; the effective mode travels with the prepared prompt to the sampling
selection and output decoder so a closed thinking prefix cannot hide an answer.
The default reasoning level is medium; explicit client choices are retained.

The port includes the upstream tool instructions, explicit/inline reasoning
deduplication, developer/system messages, and the two-tier tool-error recovery
heuristic. The warning text is the author's original text appended to the tool
observation. It is not an application-side retry loop.

Optional Jinja-only controls not present in the NInfer API (JSON tool-call format,
custom text truncation limits, preserve_reasoning alias, automatic thinking
disable with tools) are not new API features in this port. Arguments supplied as
JSON strings are parsed into objects by the existing NInfer protocol contract,
then rendered as XML parameters. Invalid non-object arguments remain errors.

The independent CTest `ninfer_froggeric_template_parity` compares rendered UTF-8
bytes against the pinned Jinja with the Hugging Face JSON filter semantics. It
also checks rewrite boundaries and initial output-channel state. Frontend tests
cover hash guards, token counting, images, and inline mode/output transitions.

Changing the Jinja file to a different upstream revision requires reviewing,
porting, and validating that revision. Its hash alone must never be added to the
registry while keeping unrelated rendering behavior.
