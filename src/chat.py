IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def format_chat(messages: list[dict], add_generation_prompt: bool = True, style: str = "qwen3.5",
                enable_thinking: bool = False) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> one prompt string.

    qwen3.5 (mirrors chat_template.jinja): no default system prompt; every message is trimmed; an assistant turn's
    text before its last </think> is reasoning (the model's own replies have no opening <think> because the prompt
    already opened it); reasoning is kept only for assistant turns after the last user query; the generation prompt
    opens a thinking block (enable_thinking) or closes an empty one. Any other style raises.
    """
    if style != "qwen3.5":
        raise ValueError(f"unknown chat style {style!r}")
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        content = m["content"].strip()
        if m["role"] == "assistant":
            reasoning = ""
            if "</think>" in content:
                reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
                content = content.split("</think>")[-1].lstrip("\n")
            if i > last_user:
                out.append(f"{IM_START}assistant\n<think>\n{reasoning}\n</think>\n\n{content}{IM_END}\n")
            else:
                out.append(f"{IM_START}assistant\n{content}{IM_END}\n")
        else:
            out.append(f"{IM_START}{m['role']}\n{content}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)
