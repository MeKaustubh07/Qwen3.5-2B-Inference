import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.cohere2.modeling_cohere2 import Cohere2DecoderLayer, Cohere2LayerNorm, Cohere2RotaryEmbedding

from streamed_hf import StreamedDecoder

HEAD_ROWS = 32768                       # rows of the tied head widened to fp32 at a time (256 MB at hidden 2048)


class StreamedCohere2(StreamedDecoder):
    """transformers' Cohere2 forward pass with one decoder layer in memory at a time."""
    layer_class = Cohere2DecoderLayer
    layer_kwargs = {"use_cache": True}

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.rotary = Cohere2RotaryEmbedding(self.config)
        self.norm = self._loaded(Cohere2LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
                                 "norm.weight")

    def position_embeddings(self, h, position_ids):
        pe = self.rotary(h, position_ids)                  # one table for every layer type
        return {t: pe for t in set(self.config.layer_types)}

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """The tied head, times logit_scale: [N, hidden] -> [N, vocab], the table widened HEAD_ROWS rows at a time."""
        return self.tied_head(h, HEAD_ROWS) * self.config.logit_scale


RAW = ["The capital of France is",
       "भारत की राजधानी नई दिल्ली है और",
       "عاصمة مصر هي",
       "中国的首都是",
       "Mji mkuu wa Kenya ni",
       "def fibonacci(n):",
       ("The history of computing is a story of abstraction. Each generation of engineers built tools that hid the "
        "details of the layer below, so the next generation could think in bigger pieces.")]
CHAT = ["What is the capital of France? Answer in one word.",
        "भारत की राजधानी क्या है? एक शब्द में उत्तर दें।"]


LONG_TOKENS_PER_LANGUAGE = 600          # 8 languages -> ~4.8K tokens, past the 4096-token window
LONG_CHUNK = 512                        # prefill tokens per pass (bounds the [heads, T, S] attention scores)


def long_rows(n: int) -> list[int]:
    """The rows whose logits the long key keeps: every 64th, the window's edge (4090-4100), the last 8."""
    return sorted(set(range(0, n, 64)) | set(range(4090, min(4101, n))) | set(range(max(0, n - 8), n)))


def main_long(d: str, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import long_texts
    tok, ref = AutoTokenizer.from_pretrained(d), StreamedCohere2(d)

    class Enc:
        @staticmethod
        def encode(text):
            return tok(text, add_special_tokens=False)["input_ids"]
    ids = tok(long_texts.build(Enc, LONG_TOKENS_PER_LANGUAGE))["input_ids"]          # BOS first: a raw prompt
    t0, cache, hs = time.perf_counter(), DynamicCache(config=ref.config), []
    for i in range(0, len(ids), LONG_CHUNK):
        hs.append(ref.run_all([ids[i:i + LONG_CHUNK]], [cache])[0])
        print(f"  {time.perf_counter() - t0:5.0f} s  prefill {min(i + LONG_CHUNK, len(ids))}/{len(ids)}", flush=True)
    h = torch.cat(hs)
    rows = long_rows(len(ids))
    logits = ref.head(h[rows])
    greedy, steps = [int(logits[-1].argmax())], []
    for _ in range(9):
        lg = ref.head(ref.run_all([[greedy[-1]]], [cache])[0][-1:])[0]
        greedy.append(int(lg.argmax())); steps.append(lg)
    torch.save({"ids": torch.tensor(ids), "final_norm": h.clone(), "rows": torch.tensor(rows), "logits": logits.clone(),
                "greedy": torch.tensor(greedy), "step_logits": torch.stack(steps),
                "tokens_per_language": LONG_TOKENS_PER_LANGUAGE, "languages": list(long_texts.SOURCES)}, out / "0.pt")
    print(f"{len(ids)} tokens, {len(rows)} logit rows -> {tok.decode(greedy)!r} in {time.perf_counter() - t0:.0f} s")


def main() -> None:
    if "--long" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--long"]
        out = Path(args[1] if len(args) > 1 else "tests/golden_aya_long")
        out.mkdir(parents=True, exist_ok=True)
        return main_long(args[0] if args else "models/tiny-aya-global", out)
    d = sys.argv[1] if len(sys.argv) > 1 else "models/tiny-aya-global"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tests/golden_aya")
    out.mkdir(parents=True, exist_ok=True)
    tok, ref = AutoTokenizer.from_pretrained(d), StreamedCohere2(d)
    prompts = [(t, False, tok(t)["input_ids"]) for t in RAW]
    prompts += [(t, True, list(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True,
                                                       add_generation_prompt=True)["input_ids"])) for t in CHAT]
    t0 = time.perf_counter()
    caps = [{} for _ in prompts]                     # chat prompts: only final_norm is kept (every position)
    keep = [len(ids) - 8 if chat else 0 for _, chat, ids in prompts]
    logits, greedy, steps = ref.greedy_all([ids for _, _, ids in prompts], 10, caps, keep,
                                    log=lambda m: print(f"  {time.perf_counter() - t0:5.0f} s  {m}", flush=True))
    for n, ((text, chat, ids), lg, gr, st, cap) in enumerate(zip(prompts, logits, greedy, steps, caps)):
        rec = {"text": text, "chat": chat, "ids": torch.tensor(ids), "greedy": torch.tensor(gr), "logits": lg.clone(),
               "logits_from": keep[n], "step_logits": st.clone(),
               **({"final_norm": cap["final_norm"]} if chat else cap)}
        torch.save(rec, out / f"{n}.pt")
        print(f"{n}: {len(ids)} tokens{' (chat)' if chat else ''} -> {tok.decode(gr)!r}", flush=True)
    print(f"{len(prompts)} prompts in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
