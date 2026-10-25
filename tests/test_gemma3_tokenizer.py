import itertools
import json
import os
import random
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "src")
from tokenizers import Tokenizer as HFTokenizer, decoders, models as hf_models, normalizers, pre_tokenizers

from chat import TemplateChat
from tokenizer import Tokenizer

MODEL, GOLDEN = "models/sarvam-translate", "tests/golden_tokens_sarvam.json"
all_ok = True


def check(name, ok):
    global all_ok
    all_ok &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def gemma_like(vocab: dict, merges: list, **bpe) -> HFTokenizer:
    hf = HFTokenizer(hf_models.BPE(vocab=vocab, merges=merges, **bpe))
    hf.normalizer = normalizers.Replace(" ", "▁")
    hf.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    hf.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    return hf


def load(spec: dict, tmp: str) -> Tokenizer:
    path = os.path.join(tmp, f"{len(os.listdir(tmp))}.json")
    json.dump(spec, open(path, "w", encoding="utf-8"))
    return Tokenizer(path)


# ---------------------------------------------------------------- 1. synthetic tables
# Gemma's vocabulary has all 256 byte tokens, so it never reaches the unknown token. Here bytes 0xB1 and 0xE2 are
# missing: "ñ" (C3 B1) and "€" (E2 82 AC) are unknown while "x" and "é" fall back to bytes, and merges join byte
# tokens and the unknown token with their neighbours. Byte-like tokens test the decoder's reading of "<0x..>".
odd = ["<0x+F>", "<0x4>", "<0xG1>", "<0x▁A>", "<0x0a>", "<0xc3>"]
tokens = ["<unk>", "a", "b", "▁", "ab", "▁a", "<0x78>a", "<0xC3><0xA9>", "<unk>b", "b<unk>"] + odd + \
    [f"<0x{b:02X}>" for b in range(256) if b not in (0xB1, 0xE2)]
vocab = {t: i for i, t in enumerate(tokens)}
merges = [("a", "b"), ("▁", "a"), ("<0x78>", "a"), ("<0xC3>", "<0xA9>"), ("<unk>", "b"), ("b", "<unk>")]
strings = ["".join(p) for n in range(1, 6) for p in itertools.product("ab xéñ€", repeat=n)]
rng = random.Random(0)
id_lists = [[rng.randrange(len(vocab)) for _ in range(rng.randint(0, 12))] for _ in range(3000)]
with tempfile.TemporaryDirectory() as tmp:
    for fallback, fuse, unk in [(True, True, "<unk>"), (True, False, "<unk>"), (True, True, None),
                                (False, True, "<unk>"), (False, False, "<unk>"), (False, False, None)]:
        hf = gemma_like(vocab, merges, byte_fallback=fallback, fuse_unk=fuse, unk_token=unk)
        ours = load(json.loads(hf.to_str()), tmp)
        bad = sum(ours.encode(s) != hf.encode(s).ids for s in strings)
        bad_dec = sum(ours.decode(x) != hf.decode(x, skip_special_tokens=False) for x in id_lists)
        check(f"synthetic byte_fallback={fallback} fuse_unk={fuse} unk={unk}: ids == HF on {len(strings)} strings "
              f"({bad} differ), decode == HF on {len(id_lists)} id lists ({bad_dec} differ)", bad == 0 == bad_dec)

    base = json.loads(gemma_like(vocab, merges, byte_fallback=True, fuse_unk=True, unk_token="<unk>").to_str())
    strip = {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    metaspace = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
    edits = {"a Strip decoder": lambda s: s["decoder"]["decoders"].append(strip),
             "a Prepend normalizer": lambda s: s.update(normalizer={"type": "Sequence", "normalizers": [
                 {"type": "Prepend", "prepend": "▁"}, s["normalizer"]]}),
             "Metaspace": lambda s: s.update(pre_tokenizer=metaspace),
             "a Split on another string": lambda s: s["pre_tokenizer"].update(pattern={"String": "a"}),
             "dropout": lambda s: s["model"].update(dropout=0.1),
             "no normalizer": lambda s: s.update(normalizer=None)}
    refused = []
    for name, edit in edits.items():
        spec = json.loads(json.dumps(base))
        edit(spec)
        try:
            load(spec, tmp)
        except NotImplementedError:
            refused.append(name)
    check(f"configurations not implemented are refused: {len(refused)}/{len(edits)} ({', '.join(refused)})",
          len(refused) == len(edits))

if not os.path.exists(f"{MODEL}/tokenizer.json"):
    print(f"SKIP Sarvam-Translate: needs {MODEL}/tokenizer.json")
    sys.exit(0 if all_ok else 1)

# ---------------------------------------------------------------- 2. the real tokenizer
from transformers import AutoTokenizer

t0 = time.perf_counter()
tok = Tokenizer(f"{MODEL}/tokenizer.json")
t_load = time.perf_counter() - t0
hf = HFTokenizer.from_file(f"{MODEL}/tokenizer.json")
auto = AutoTokenizer.from_pretrained(MODEL)
n = tok.vocab_size()
check(f"vocab_size {n} == 262,145, ids contiguous, BOS {tok.bos_id}, <start_of_turn> {tok.special['<start_of_turn>']}, "
      f"<end_of_turn> {tok.special['<end_of_turn>']}, {len(tok.fallback_byte)} byte tokens; loaded in {t_load:.2f} s",
      n == 262145 and sorted(tok.id_to_token) == list(range(n)) and tok.bos_id == 2 and len(tok.fallback_byte) == 256
      and (tok.special["<start_of_turn>"], tok.special["<end_of_turn>"], tok.special["<eos>"]) == (105, 106, 1))

if not os.path.exists(GOLDEN):                         # the model is here but its answer key was never made
    print(f"FAIL Sarvam-Translate: run scripts/.venv/bin/python scripts/golden_tokens.py {MODEL} {GOLDEN}")
    sys.exit(1)
cases = json.load(open(GOLDEN, encoding="utf-8"))
failed = []
for c in cases:
    plain = tok.encode(c["text"])
    if plain != c["ids_plain"] or tok.encode(c["text"], add_bos=True) != c["ids"] or tok.decode(plain) != c["decoded"]:
        failed.append(c["text"])
for t in failed[:5]:
    print("  differs:", repr(t[:60]))
n_fallback = sum(any(tok.fallback_byte.get(i) is not None for i in c["ids_plain"]) for c in cases)
check(f"answer key: {len(cases) - len(failed)}/{len(cases)} cases match ids, ids with BOS and decode "
      f"({n_fallback} with byte fallback, longest {max(len(c['text']) for c in cases):,} characters)", not failed)

auto_bad = sum(auto.encode(c["text"], add_special_tokens=False) != c["ids_plain"] or auto.encode(c["text"]) != c["ids"]
               or auto.decode(c["ids_plain"]) != c["decoded"] for c in cases)
check(f"transformers' {type(auto).__name__} gives the answer key's ids and decode on all {len(cases)} cases "
      f"({auto_bad} differ)", auto_bad == 0)

rng = random.Random(1)
chars = sorted({ch for c in cases for ch in c["text"]})
pool = chars + ["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "\n\n", "\t\t", "  ", "▁▁", "<0xEA>", "<td>",
                "\r\n"]
texts = ["".join(rng.choice(pool) for _ in range(rng.randint(0, 80))) for _ in range(3000)]
bad = [t for t in texts if tok.encode(t) != hf.encode(t, add_special_tokens=False).ids]
check(f"ids == HF tokenizers' on {len(texts)} random texts over the key's {len(chars)} characters and added "
      f"tokens ({len(bad)} differ)", not bad)
golden_ids, byte_ids = [i for c in cases for i in c["ids_plain"]], sorted(tok.fallback_byte)
draw = [lambda: rng.choice(byte_ids), lambda: rng.choice(golden_ids), lambda: rng.randrange(n)]
id_lists = [[rng.choice(draw)() for _ in range(rng.randint(0, 24))] for _ in range(3000)]
bad = sum(tok.decode(x) != hf.decode(x, skip_special_tokens=False) for x in id_lists)
check(f"decode == HF on {len(id_lists)} random id lists, a third of them byte tokens ({bad} differ)", bad == 0)

# ---------------------------------------------------------------- 3. speed: a text between added tokens is one chunk
long_case = max(cases, key=lambda c: len(c["text"]))["text"]
blocks = [(0x0900, 0x0980), (0x0B80, 0x0C00), (0x0D00, 0x0D80), (0xABC0, 0xAC00), (0x0600, 0x0700), (0x4E00, 0x9FA6)]
mixed = "".join(chr(rng.randrange(*rng.choice(blocks))) for _ in range(80000))


def best_of_3(text):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        tok.encode(text)
        times.append(time.perf_counter() - t0)
    return min(times)


t_long, t20, t80 = best_of_3(long_case), best_of_3(mixed[:20000]), best_of_3(mixed)
t0 = time.perf_counter()
hf.encode(long_case, add_special_tokens=False)
t_hf = time.perf_counter() - t0
same = tok.encode(mixed) == hf.encode(mixed, add_special_tokens=False).ids
check(f"the key's {len(long_case):,}-character unspaced case in {t_long * 1e3:.1f} ms (< 250; HF {t_hf * 1e3:.1f} ms); "
      f"80,000 random Indic / Meitei / Arabic / CJK characters == HF in {t80 * 1e3:.0f} ms, {t80 / t20:.1f}x the "
      f"first 20,000 (n log n: ~4.6x; quadratic: 16x)", t_long < 0.25 and same and t80 / t20 < 8)

# ---------------------------------------------------------------- 4. chat template
chat = TemplateChat(f"{MODEL}/tokenizer_config.json")
system = {"role": "system", "content": "Translate the text below to Hindi."}
conversations = {
    "one user turn": [{"role": "user", "content": "Be the change you wish to see in the world."}],
    "system + user (the card's prompt)": [system, {"role": "user", "content": "  The weather is pleasant today.\n"}],
    "multi-turn": [system, {"role": "user", "content": "Good morning"}, {"role": "assistant", "content": "सुप्रभात"},
                   {"role": "user", "content": "ꯃꯤꯇꯩꯂꯣꯟ ꯑꯁꯤ ꯑꯩꯒꯤ ꯃꯃꯥ ꯂꯣꯟꯅꯤ꯫"}],
}
for name, messages in conversations.items():
    ours = chat.render(messages)
    want = auto.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(ours)
    check(f"chat, {name}: rendering == apply_chat_template ({len(want)} characters); encode == HF's "
          f"({len(ids)} ids, one BOS written by the template)",
          ours == want and ids == hf.encode(want, add_special_tokens=False).ids
          == auto.encode(want, add_special_tokens=False) and ids[0] == 2 and ids[1] != 2)

bad_order = [system, {"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Hello"}]


def raised(fn) -> Exception | None:
    try:
        fn()
    except Exception as e:
        return e
    return None


ours_err = raised(lambda: chat.render(bad_order))
hf_err = raised(lambda: auto.apply_chat_template(bad_order, tokenize=False, add_generation_prompt=True))
check(f"chat, roles out of order: refused by both (ours {type(ours_err).__name__}, transformers "
      f"{type(hf_err).__name__}) with the template's message ({str(hf_err)!r})",
      isinstance(ours_err, ValueError) and hf_err is not None and str(ours_err) == str(hf_err))

# ---------------------------------------------------------------- 5. streaming
# ByteFallback decodes a run of byte tokens all or nothing, so decode(a + b) need not start with decode(a): a stream
# that sent a character from a run cannot take it back when the run goes on with bytes that make it invalid.
import types

import torch

from generate import generate_stream
from sampler import SamplingParams
from server.metrics import Metrics
from server.scheduler import Request, Scheduler


def server_stream(ids: list[int]) -> list[str]:
    s = Scheduler.__new__(Scheduler)                   # the detokenizer alone: all that _emit and _finish read
    s.eng, s.tok, s.metrics = types.SimpleNamespace(eos_ids=set()), tok, Metrics()
    s.running, s.prefilling, s.active = [], [], set()
    r = Request("r", [], SamplingParams(), len(ids))
    for t in ids:
        s._emit(r, t)
    return [v for kind, v in (r.out.get() for _ in range(r.out.qsize())) if kind == "token"]


class Scripted:                                        # a model whose next token is always the script's
    def __init__(self, script: list[int]):
        self.script = iter(script)

    def new_state(self, max_len):
        return None

    def forward(self, ids, state=None, last_only=False):
        nxt = next(self.script)
        return torch.nn.functional.one_hot(torch.tensor([nxt]), nxt + 1).float()


def cli_stream(ids: list[int]) -> list[str]:
    greedy = SamplingParams(temperature=0, repetition_penalty=1.0)
    return list(generate_stream(Scripted(ids), tok, [tok.bos_id], greedy, len(ids), set()))


byte = [tok.vocab[f"<0x{b:02X}>"] for b in range(256)]
cut = [byte[0xEA], byte[0xAF], byte[0x87], byte[0xEA], byte[0xAF]] + tok.encode(" next")   # ꯇ, then a cut ꯇ
stray = [byte[0xEA], byte[0xAF], byte[0x87]] * 2 + [byte[0x80]] + tok.encode(" x")         # ꯇꯇ, then a stray byte
scripts = [(0x0900, 0x0D80), (0xABC0, 0xAC00), (0x1C50, 0x1C80), (0x13000, 0x13430), (0x1F600, 0x1F650)]


def piece() -> list[int]:
    """A character's byte tokens, whole or cut short, a stray continuation byte, or an id from the answer key."""
    kind, c = rng.randrange(4), chr(rng.randrange(*rng.choice(scripts))).encode("utf-8")
    if kind < 2:
        return [byte[x] for x in (c if kind == 0 else c[:rng.randrange(1, len(c))])]
    return [byte[rng.randrange(0x80, 0xC0)]] if kind == 2 else [rng.choice(golden_ids)]


lists = [cut, stray] + [[t for _ in range(rng.randint(1, 8)) for t in piece()] for _ in range(300)]
want = [hf.decode(x, skip_special_tokens=False) for x in lists]
bad_server = sum("".join(server_stream(x)) != w for x, w in zip(lists, want))
bad_cli = sum("".join(cli_stream(x)) != w for x, w in zip(lists, want))
check(f"streams == HF's decode, server (_emit) and generate_stream, on {len(lists)} id lists of whole, cut and stray "
      f"UTF-8 byte tokens ({bad_server} and {bad_cli} differ); a character then a cut one streams as "
      f"{server_stream(cut)} (HF {want[0]!r}), two then a stray byte as {server_stream(stray)} (HF {want[1]!r})",
      bad_server == 0 == bad_cli)

sys.exit(0 if all_ok else 1)
