import os
import resource
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from chat import TemplateChat, format_chat
from config import Gemma3Config
from golden_gemma3 import messages
from models.gemma3 import Gemma3Model
from tokenizer import Tokenizer
from weight_loader import open_weights

D, KEY = "models/sarvam-translate", "tests/golden_gemma3_long/0.pt"
CHUNK = 600                                                # chunks 0-600, 600-1200 (across the window's edge), 1200-
if not os.path.exists(f"{D}/model.safetensors.index.json"):
    print(f"SKIP: needs the checkpoint in {D}")
    sys.exit(0)
if not os.path.exists(KEY):
    print("FAIL: no long answer key: run scripts/.venv/bin/python scripts/golden_gemma3.py --long")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def rel_rows(a, b):
    """Each row's worst error over that row's largest value -> per-row errors [rows]."""
    return (a - b).abs().amax(-1) / b.abs().amax(-1)


t0 = time.perf_counter()
key = torch.load(KEY)
ids, n, rows, greedy = key["ids"], len(key["ids"]), key["rows"].tolist(), key["greedy"]
tok, tmpl = Tokenizer(f"{D}/tokenizer.json"), TemplateChat(f"{D}/tokenizer_config.json")
cfg = Gemma3Config.from_json(f"{D}/config.json")
model = Gemma3Model(cfg, open_weights(f"{D}/model.safetensors.index.json"), stream=True)
W = cfg.sliding_window

# 1. the engine's tokenizer and chat template give the key's ids (made with transformers' tokenizer and template)
check(f"ids from the engine's tokenizer and chat template == the key's for the {n}-token prompt "
      f"({key['source']}, into {key['target']})",
      tok.encode(format_chat(messages(key["target"], key["text"]), style=tmpl)) == ids.tolist())

# 2. prefill in CHUNK-token chunks into one contiguous fp32 state; every layer kept at the key's rows. A copy of the
# state is kept at the first chunk boundary (before the window binds) for the sensitivity check.
st, hs, fork = model.new_state(n + len(greedy) + 1), [], None
layers = {k: [] for k in key["layers"]}
for i in range(0, n, CHUNK):
    if i == CHUNK:
        fork = st.fork()
    cap = {}
    hs.append(model.packed_hidden([(ids[i:i + CHUNK], st)], capture=cap))
    mine = [r - i for r in rows if i <= r < i + CHUNK]
    for k in layers:
        layers[k].append(cap[k][mine])
    del cap
    print(f"      {time.perf_counter() - t0:5.0f} s  prefill {min(i + CHUNK, n)}/{n}", flush=True)
h = torch.cat(hs)
err = rel_rows(h, key["final_norm"])
before, after = err[:W].max().item(), err[W:].max().item()
check(f"final norm of all {n} positions == the key's: worst row {before:.1e} before position {W}, {after:.1e} from it "
      f"on (median {err.median().item():.1e})", max(before, after) < 1e-4)
layers = {k: torch.cat(v) for k, v in layers.items()}
lerr = {k: rel_rows(v, key["layers"][k]).max().item() for k, v in layers.items()}
worst = max(lerr, key=lerr.get)
check(f"the embedding and all {cfg.num_hidden_layers} layers at the key's {len(rows)} rows (every 64th, "
      f"{W - 6}-{W + 6}, the last 8) == the key's: embedding bit-identical, worst row {lerr[worst]:.1e} ({worst})",
      torch.equal(layers["embed"], key["layers"]["embed"]) and lerr[worst] < 1e-4)
logits = model.head(h[rows])
logit_err = rel_rows(logits, key["logits"]).max().item()
check(f"logits at the key's {len(rows)} rows: worst row {logit_err:.1e}", logit_err < 1e-4)

# 3. decode steps on the key's own greedy tokens (one batched decode step each, as the server runs them), then greedy
steps = torch.cat([model.decode_batch([t], [st]) for t in greedy[:-1].tolist()])
serr = rel_rows(steps, key["step_logits"]).max().item()
check(f"{len(steps)} decode steps at positions {n}-{n + len(steps) - 1} == the key's, worst row {serr:.1e}",
      serr < 1e-4)
first = int(logits[-1].argmax())
check(f"greedy: the first new token == the key's ({first}), and the key's {len(greedy)} tokens are the argmax at "
      f"every step: {tok.decode(greedy.tolist())!r}",
      first == int(greedy[0]) and torch.equal(steps.argmax(-1), greedy[1:]))
del steps

# 4. sensitivity: from position CHUNK with the window off, the rows before W stay on the key, the rows from W leave it
real = model._window
model._window = lambda i: None
off = model.packed_hidden([(ids[CHUNK:], fork)])
model._window = real
on_err = rel_rows(off[:W - CHUNK], key["final_norm"][CHUNK:W])
off_err = rel_rows(off[W - CHUNK:], key["final_norm"][W:])
check(f"window off from position {CHUNK}: positions {CHUNK}-{W - 1} unchanged (worst row {on_err.max().item():.1e}); "
      f"from {W} on worst row {off_err.max().item():.1e} (median {off_err.median().item():.1e}), vs {after:.1e} "
      f"with it", on_err.max().item() < 1e-4 and off_err.max().item() > 100 * after)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30                          # bytes on macOS
print(f"      {time.perf_counter() - t0:.0f} s, peak RSS {peak:.2f} GiB")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
