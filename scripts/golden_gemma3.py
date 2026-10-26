import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RMSNorm, Gemma3RotaryEmbedding

from streamed_hf import StreamedDecoder, long_key, long_rows, provenance

STOP = (1, 106)                         # <eos>, <end_of_turn> (generation_config.json)
N_NEW = 40                              # greedy tokens per prompt at most
SYSTEM = "Translate the text below to {}."           # the model card's prompt


class StreamedGemma3(StreamedDecoder):
    """transformers' Gemma 3 text forward pass with one decoder layer in memory at a time."""
    layer_class = Gemma3DecoderLayer

    def __init__(self, model_dir: str, **kw):
        super().__init__(model_dir, prefixes=("language_model.model.", "model."), **kw)
        c = self.config
        if c.final_logit_softcapping is not None:
            raise ValueError(f"final_logit_softcapping {c.final_logit_softcapping}: the answer key has no softcap")
        if c.use_bidirectional_attention:
            raise ValueError("use_bidirectional_attention: the answer key builds causal masks only")
        self.rotary = Gemma3RotaryEmbedding(c)
        self.norm = self._loaded(Gemma3RMSNorm(c.hidden_size, eps=c.rms_norm_eps), "norm.weight")
        self.embed_scale = torch.tensor(c.hidden_size ** 0.5)          # fp32: Gemma3TextScaledWordEmbedding's buffer

    def embed(self, ids: list[int]) -> torch.Tensor:
        return super().embed(ids) * self.embed_scale

    def position_embeddings(self, h, position_ids):
        return {t: self.rotary(h, position_ids, t) for t in set(self.config.layer_types)}

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.tied_head(h)


TRANSLATE = [  # (target language, source text)
    ("Hindi", "The library opens at nine in the morning, and students often wait outside before the doors open."),
    ("Marathi", "Farmers in the village planted rice early this year because the monsoon rains arrived on time."),
    ("Tamil", "Please drink plenty of water during the summer and avoid walking in the afternoon sun."),
    ("Bengali", "My grandmother tells us stories about her childhood every evening after dinner."),
    ("Urdu", "The train to Lucknow was delayed by two hours because of thick fog near the station."),
    ("Manipuri", "The festival brings the whole community together for music, dance and food."),
    ("Santali", "Children walk to the school near the forest every morning with their friends."),
    ("English", "मानसून की पहली बारिश के बाद पूरे शहर में हरियाली छा गई और लोग खुशी से बाहर निकल आए।"),
    ("English", "आमच्या शाळेत दरवर्षी विज्ञान प्रदर्शन भरते आणि विद्यार्थी आपले प्रकल्प अभिमानाने दाखवतात."),
]
RAW = ["Meiteilon is written in its own script, Meitei Mayek: ꯃꯤꯇꯩꯂꯣꯟ ꯑꯁꯤ ꯑꯩꯒꯤ ꯃꯃꯥ ꯂꯣꯟꯅꯤ꯫ In English, this sentence "
       "means"]

LONG_LANGUAGE = "hi"                    # scripts/long_texts.py: Andher Nagari (Bharatendu Harishchandra, 1881)
LONG_TOKENS = 1400                      # tokens of its text; with the template ~1.42K, past the 1024-token window
LONG_NEW = 16


def messages(target: str, text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM.format(target)}, {"role": "user", "content": text}]


def chat_ids(tok, target: str, text: str) -> list[int]:
    return list(tok.apply_chat_template(messages(target, text), tokenize=True, add_generation_prompt=True)["input_ids"])


def gemma3_long_key(ref: StreamedGemma3, ids: list[int], n_new: int, log=None) -> dict:
    """The --long record: one prefill pass, every layer kept at the rows, greedy until a stop id or n_new tokens."""
    return long_key(ref, ids, n_new, long_rows(len(ids), ref.config.sliding_window), layers=True, stop=STOP, log=log)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30          # bytes on macOS


def done(path: Path, ids: list[int], n_new: int, made: dict, force: bool) -> bool:
    """path holds a finished record for these ids and this many new tokens at most, made as `made` describes."""
    if force or not path.exists():
        return False
    rec = torch.load(path, mmap=True)
    return rec["ids"].tolist() == ids and rec.get("n_new") == n_new and rec.get("provenance") == made


def main_long(d: str, out: Path, force: bool) -> None:
    import long_texts
    tok, made = AutoTokenizer.from_pretrained(d), provenance(d)
    text = long_texts.parts(lambda t: tok(t, add_special_tokens=False)["input_ids"], LONG_TOKENS, (LONG_LANGUAGE,))[0]
    ids = chat_ids(tok, "English", text)
    if done(out / "0.pt", ids, LONG_NEW, made, force):
        return print(f"{out / '0.pt'} is already done")
    ref = StreamedGemma3(d)
    if len(ids) <= ref.config.sliding_window + 128:
        raise SystemExit(f"{len(ids)} tokens: not far enough past the {ref.config.sliding_window}-token window")
    t0 = time.perf_counter()
    rec = gemma3_long_key(ref, ids, LONG_NEW, log=lambda m: print(f"  {time.perf_counter() - t0:5.0f} s  {m}", flush=True))
    rec |= {"text": text, "target": "English", "source": long_texts.SOURCES[LONG_LANGUAGE][1], "n_new": LONG_NEW,
            "provenance": made}
    torch.save(rec, out / "0.pt")
    print(f"{len(ids)} tokens, {len(rec['rows'])} rows -> {tok.decode(rec['greedy'])!r}")
    print(f"{time.perf_counter() - t0:.0f} s, peak RSS {peak_rss_gib():.2f} GiB")


CALIB_GROUP = 22                        # prompts per pass over the layers (their fp32 KV caches: ~0.6 GB)


def main_calib(d: str, out: Path) -> None:
    import json
    import numpy as np
    import calib_texts
    import lang_kl
    tok, prompts = AutoTokenizer.from_pretrained(d), calib_texts.load_prompts()
    n_tok = len(tok)                                                   # 262145 ids (the head has 262208 rows)
    for p in prompts:
        if chat_ids(tok, calib_texts.NAMES[p["tgt"]], p["source"]) != p["ids"]:
            raise SystemExit(f"{p['lang']} {p['split']} {p['src']}->{p['tgt']}: ids differ from apply_chat_template")
    index = [{k: p[k] for k in ("lang", "src", "tgt", "split", "logits_from")} | {"ids": p["ids"] + p["output"],
             "file": f"{i}.npy"} for i, p in enumerate(prompts)]
    old = json.load(open(out / "index.json")) if (out / "index.json").exists() else []
    todo = [i for i, r in enumerate(index) if not (i < len(old) and old[i] == r and (out / r["file"]).exists())]
    json.dump(index, open(out / "index.json", "w"), ensure_ascii=False)
    print(f"{len(index) - len(todo)} of {len(index)} prompts already done")
    if not todo:
        return
    ref, t0 = StreamedGemma3(d), time.perf_counter()
    for g in range(0, len(todo), CALIB_GROUP):
        group = todo[g:g + CALIB_GROUP]
        hs = ref.run_all([index[i]["ids"] for i in group], [DynamicCache(config=ref.config) for _ in group])
        for i, h in zip(group, hs):
            r = index[i]
            np.save(out / r["file"], lang_kl.encode_rows(ref.head(h[r["logits_from"]:-1])[:, :n_tok].numpy()))
        print(f"  {time.perf_counter() - t0:5.0f} s  {g + len(group)}/{len(todo)} prompts, "
              f"{sum(len(index[i]['ids']) for i in group)} tokens", flush=True)
    print(f"{len(todo)} prompts in {time.perf_counter() - t0:.0f} s, peak RSS {peak_rss_gib():.2f} GiB")


def main() -> None:
    args, force = [a for a in sys.argv[1:] if a not in ("--long", "--calib", "--force")], "--force" in sys.argv
    d = args[0] if args else "models/sarvam-translate"
    if "--calib" in sys.argv:
        out = Path(args[1] if len(args) > 1 else "models/sarvam-translate-calib/ref")
        out.mkdir(parents=True, exist_ok=True)
        return main_calib(d, out)
    if "--long" in sys.argv:
        out = Path(args[1] if len(args) > 1 else "tests/golden_gemma3_long")
        out.mkdir(parents=True, exist_ok=True)
        return main_long(d, out, force)
    out = Path(args[1] if len(args) > 1 else "tests/golden_gemma3")
    out.mkdir(parents=True, exist_ok=True)
    tok, made = AutoTokenizer.from_pretrained(d), provenance(d)
    prompts = [(text, target, chat_ids(tok, target, text)) for target, text in TRANSLATE]
    prompts += [(text, None, tok(text)["input_ids"]) for text in RAW]
    todo = [n for n, (*_, ids) in enumerate(prompts) if not done(out / f"{n}.pt", ids, N_NEW, made, force)]
    if len(todo) < len(prompts):
        print(f"{len(prompts) - len(todo)} of {len(prompts)} prompts already done")
    if not todo:
        return
    ref, t0 = StreamedGemma3(d), time.perf_counter()
    caps = [{} for _ in todo]

    def save(j, lg, gr, st):
        n, (text, target, ids) = todo[j], prompts[todo[j]]
        torch.save({"text": text, "target": target, "chat": target is not None, "ids": torch.tensor(ids),
                    "n_new": N_NEW, "provenance": made, "greedy": torch.tensor(gr), "logits": lg.clone(), "step_logits": st.clone(),
                    **caps[j]}, out / f"{n}.pt")
        print(f"{n}: {len(ids)} tokens{f' -> {target}' if target else ' (raw)'}: {len(gr)} new "
              f"{'(stopped)' if gr[-1] in STOP else '(no stop)'} {tok.decode(gr)!r}", flush=True)
    ref.greedy_all([prompts[n][2] for n in todo], N_NEW, caps, stop=STOP, on_done=save,
                   log=lambda m: print(f"  {time.perf_counter() - t0:5.0f} s  {m}", flush=True))
    print(f"{len(todo)} prompts in {time.perf_counter() - t0:.0f} s, peak RSS {peak_rss_gib():.2f} GiB")


if __name__ == "__main__":
    main()
