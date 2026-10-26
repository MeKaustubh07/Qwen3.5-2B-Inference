import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
from engine import load_engine
from quant import QtFile, QuantTensor, load_policy, scheme_for
from serving_targets import free_gpu, gpu_peak, kl, metal_aborts, top2_gap

D, G, LONG = "models/sarvam-translate", "tests/golden_gemma3", "tests/golden_gemma3_long/0.pt"
POLICY = "configs/quant/sarvam-translate.json"
BUILDS = {"int8": ("metal-int8", "model.int8.qt", ""), "int4-flat": ("metal-int4", "model.int4-flat.qt", ""),
          "int4": ("metal-int4", "model.int4.qt", f" --policy {POLICY}")}
# KL on the main key, KL on the long key, top-1 flips (share of non-tied positions), each a margin over what the build
# measured (KL main / long, flips): INT8 0.0006 / 0.0016, 0/483; flat INT4 0.0509 / 0.0731, 19; policy 0.0413 / 0.0522, 4
GATES = {"int8": (0.005, 0.005, 0.01), "int4-flat": (0.065, 0.09, 0.06), "int4": (0.05, 0.065, 0.02)}
POLICY_GAIN = 0.9                                              # the policy's KL at most this times flat INT4's
if not os.path.exists(f"{D}/model.safetensors.index.json") or not torch.backends.mps.is_available():
    print(f"SKIP: needs the checkpoint in {D} and a Metal GPU")
    sys.exit(0)
goldens = sorted((f for f in os.listdir(G) if f.endswith(".pt")), key=lambda f: int(f[:-3])) if os.path.isdir(G) else []
if not goldens or not os.path.exists(LONG):
    print("FAIL: no answer keys: run scripts/.venv/bin/python scripts/golden_gemma3.py and golden_gemma3.py --long")
    sys.exit(1)

torch.set_grad_enabled(False)
results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


t0, started = time.perf_counter(), time.strftime("%Y-%m-%d %H:%M:%S")
recs = [torch.load(f"{G}/{f}") for f in goldens]
key_long = torch.load(LONG)


def measure(m, tok, eos_ids) -> dict:
    """The model's next-token distributions vs the main key's at every recorded position (fed the key's tokens), and
    free greedy per prompt from a fork of the same prefill: every live sequence of both in one decode_batch per step."""
    V = tok.vocab_size()
    template = set(tok.special.values()) - {tok.bos_id}
    keys = [r["greedy"].tolist() for r in recs]
    forced = [m.new_state(len(r["ids"]) + len(g) + 1) for r, g in zip(recs, keys)]
    ours = [[m.forward(r["ids"], state=st).cpu()] for r, st in zip(recs, forced)]
    free = [st.fork() for st in forced]
    nxt, same, alive = [int(o[0][-1, :V].argmax()) for o in ours], [0] * len(recs), [True] * len(recs)
    while True:
        for j, g in enumerate(keys):                           # free greedy: how many tokens equal the key's
            if alive[j]:
                alive[j] = nxt[j] == g[same[j]]
                same[j] += alive[j]
                alive[j] &= same[j] < len(g)
        teach = [j for j, g in enumerate(keys) if len(ours[j]) < len(g)]
        run = [j for j in range(len(recs)) if alive[j]]
        if not teach and not run:
            break
        out = m.decode_batch([keys[j][len(ours[j]) - 1] for j in teach] + [nxt[j] for j in run],
                             [forced[j] for j in teach] + [free[j] for j in run]).cpu()
        for j, row in zip(teach, out):
            ours[j].append(row[None])
        for j, row in zip(run, out[len(teach):]):
            nxt[j] = int(row[:V].argmax())
    del forced, free
    kls, served, per = [], [], []
    for r, g, o, sm in zip(recs, keys, ours, same):
        n, o, ref = len(r["ids"]), torch.cat(o), torch.cat([r["logits"], r["step_logits"]])
        kls.append(kl(ref, o, V))
        seq = r["ids"].tolist() + g                            # seq[i]: the token position i reads
        stop = next((n + k for k, t in enumerate(g) if t in eos_ids), len(seq))
        served.append(torch.tensor([i < stop and (i >= n - 1 or seq[i] not in template) for i in range(len(o))]))
        gap = top2_gap(ref, V)
        clear = gap > 0.5
        flips = int(((o[:, :V].argmax(-1) != ref[:, :V].argmax(-1)) & clear).sum())
        per.append(dict(name=(r["target"] or "raw")[:3], kl=kls[-1].mean().item(), flips=flips,
                        clear=int(clear.sum()), same=sm, n=len(g), gap=gap[n - 1 + sm].item() if sm < len(g) else None))
    k, is_served = torch.cat(kls), torch.cat(served)
    return dict(kl=k.mean().item(), kl_max=k.max().item(), kl_served=k[is_served].mean().item(),
                n_served=int(is_served.sum()), flips=sum(p["flips"] for p in per), clear=sum(p["clear"] for p in per),
                greedy=[(p["same"], p["n"]) for p in per], per=per, n=len(k))


def measure_long(m, V, W) -> dict:
    """The long key: 512-token chunks into one state, the key's rows and decode steps."""
    ids, rows, g = key_long["ids"], key_long["rows"], key_long["greedy"].tolist()
    st = m.new_state(len(ids) + len(g) + 1)
    h = torch.cat([m.packed_hidden([(ids[i:i + 512], st)]) for i in range(0, len(ids), 512)])
    lg = m.head(h[rows]).float().cpu()
    stp = torch.cat([m.forward(torch.tensor([t]), state=st) for t in g[:-1]]).float().cpu()
    k_rows, k_steps = kl(key_long["logits"], lg, V), kl(key_long["step_logits"], stp, V)
    agree = int(lg[-1, :V].argmax() == g[0]) + int((stp[:, :V].argmax(-1) == key_long["greedy"][1:]).sum())
    return dict(before=k_rows[rows < W].mean().item(), after=k_rows[rows >= W].mean().item(),
                steps=k_steps.mean().item(), all=torch.cat([k_rows, k_steps]).mean().item(), agree=agree, n=len(g))


def decode_speed(m, tok) -> float:
    """Single-stream decode, tokens per second: the median of 3 runs of 20 steps."""
    st, runs = m.new_state(128), []
    m.forward(torch.tensor(tok.encode("The capital of France is", add_bos=True)), state=st, last_only=True)
    for _ in range(3):
        m.forward(torch.tensor([12]), state=st, last_only=True)
    for _ in range(3):
        m.b.sync(); t = time.perf_counter()
        for _ in range(20):
            int(m.forward(torch.tensor([12]), state=st, last_only=True)[0].argmax())
        runs.append(20 / (time.perf_counter() - t))
    return sorted(runs)[1]


def schemes_as_policy(path, keep) -> tuple[int, bool]:
    """(INT8 tensors in the .qt, every tensor's scheme == what quantize.py --policy gives it with these groups)."""
    q = QtFile(path)
    got = {n: w.scheme for n in q.tensor_names() if isinstance(w := q.get(n), QuantTensor)}
    return sum(s == "int8" for s in got.values()), all(s == scheme_for(n, "int4", keep) for n, s in got.items())


summary, peaks = {}, {}
for build, (backend, file, how) in BUILDS.items():
    qt = f"{D}/{file}"
    if not os.path.exists(qt):
        check(f"{build}: {qt} exists (scripts/quantize.py {D}/model.safetensors.index.json {qt} --scheme "
              f"{backend.split('-')[1]} --prefix language_model.model.{how})", False)
        continue
    if build == "int4":
        keep = load_policy(POLICY) if os.path.exists(POLICY) else None
        n8, same = schemes_as_policy(qt, keep) if keep is not None else (0, False)
        check(f"int4: {qt} is quantized with the policy {POLICY} ({len(keep or ())} groups INT8; "
              f"{n8} INT8 tensors, the tied embedding included: every tensor's scheme as the policy gives it {same})",
              same and n8 > 1)
    eng = load_engine("sarvam-translate", backend, qt)
    m, tok = eng.model, eng.tokenizer
    V, W = tok.vocab_size(), m.config.sliding_window
    shipped = m.kv_dtype                                       # the registry's: bf16
    for kv in (torch.float32, torch.bfloat16):
        m.kv_dtype = kv
        free_gpu()
        torch.accelerator.reset_peak_memory_stats()
        s = summary[build, kv] = measure(m, tok, eng.eos_ids) | {"long": measure_long(m, V, W)}
        peaks[build, kv] = gpu_peak()
        s["tps"] = decode_speed(m, tok)
        x = s["long"]
        print(f"      {build}, {str(kv)[6:]} KV: {s['n']} positions | KL vs fp32 mean {s['kl']:.4f} (max "
              f"{s['kl_max']:.3f}; at the {s['n_served']} served positions {s['kl_served']:.4f}) | top-1 flips "
              f"{s['flips']}/{s['clear']} non-tied | greedy == key for {sum(a for a, _ in s['greedy'])}/"
              f"{sum(b for _, b in s['greedy'])} tokens | decode {s['tps']:.1f} tok/s | long: rows before {W} "
              f"{x['before']:.4f}, from it {x['after']:.4f}, decode steps {x['steps']:.4f}, greedy {x['agree']}/"
              f"{x['n']} | GPU {peaks[build, kv] / 2**30:.2f} GiB\n        per prompt (KL, flips, greedy): "
              + "; ".join(f"{p['name']} {p['kl']:.3f} {p['flips']}/{p['clear']} {p['same']}/{p['n']}"
                          + (f" (gap {p['gap']:.2f})" if p["gap"] is not None else "") for p in s["per"]),
              flush=True)
    if build == "int4":                                        # the served build: the embedding scale it serves
        m.kv_dtype = shipped
        served = m.embed_scale
        m.embed_scale = torch.tensor(m.config.hidden_size ** 0.5).item()            # fp32, as the key
        summary["fp32-scale"] = measure(m, tok, eng.eos_ids)
        m.embed_scale = served
    m.kv_dtype = shipped
    gb = sum(w.nbytes if isinstance(w, QuantTensor) else w.numel() * w.element_size() for w in m._cache.values()) / 1e9
    print(f"      {build}: weights resident {gb:.2f} GB", flush=True)
    del eng, m
    free_gpu()

for (build, kv), s in ((k, v) for k, v in summary.items() if isinstance(k, tuple)):   # every run: a broken one must
    kl_max, long_max, flips_max = GATES[build]                                        # not hide behind the other
    check(f"{build}, {str(kv)[6:]} KV: KL vs the fp32 key < {kl_max} ({s['kl']:.4f}), long key < {long_max} "
          f"({s['long']['all']:.4f}), top-1 flips <= {flips_max:.0%} of non-tied positions ({s['flips']}/{s['clear']})",
          s["kl"] < kl_max and s["long"]["all"] < long_max and s["flips"] <= flips_max * s["clear"])
for build in BUILDS:
    if (build, torch.float32) in summary and (build, torch.bfloat16) in summary:
        d = summary[build, torch.bfloat16]["kl"] - summary[build, torch.float32]["kl"]
        dl = summary[build, torch.bfloat16]["long"]["all"] - summary[build, torch.float32]["long"]["all"]
        check(f"{build}: bf16 KV changes the KL by at most 0.01 either way ({d:+.4f}; long key {dl:+.4f})",
              abs(d) <= 0.01 and abs(dl) <= 0.01)
for kv in (torch.float32, torch.bfloat16):                  # INT8 may leave the key's greedy path only at a near-tie
    if ("int8", kv) in summary:
        per = summary["int8", kv]["per"]
        off = [p for p in per if p["gap"] is not None]
        check(f"int8, {str(kv)[6:]} KV: free greedy == the key on {len(per) - len(off)}/{len(per)} prompts; the others "
              f"leave it where the key's top two are within 0.5 ("
              + (", ".join(f"{p['name']} at token {p['same']}: {p['gap']:.2f}" for p in off) or "none") + ")",
              all(p["gap"] <= 0.5 for p in off))
if ("int4", torch.bfloat16) in summary and ("int4-flat", torch.bfloat16) in summary:
    p, f = summary["int4", torch.bfloat16], summary["int4-flat", torch.bfloat16]
    check(f"the calibrated policy, measured on text it never saw, at most {POLICY_GAIN} x flat INT4's KL: "
          f"{p['kl']:.4f} vs {f['kl']:.4f} ({p['kl'] / f['kl']:.2f}x; served positions {p['kl_served']:.4f} vs "
          f"{f['kl_served']:.4f}); long key {p['long']['all']:.4f} vs {f['long']['all']:.4f} "
          f"({p['long']['all'] / f['long']['all']:.2f}x)",
          p["kl"] <= POLICY_GAIN * f["kl"] and p["long"]["all"] <= POLICY_GAIN * f["long"]["all"])
if "fp32-scale" in summary:
    b, f = summary["int4", torch.bfloat16], summary["fp32-scale"]
    check(f"int4, bf16 KV: the served embedding scale 50.5 (bf16) vs 50.596 (fp32, the key's): KL vs the key "
          f"{b['kl']:.4f} vs {f['kl']:.4f} ({b['kl'] - f['kl']:+.4f}; served positions "
          f"{b['kl_served'] - f['kl_served']:+.4f}); "
          f"greedy tokens == key {sum(a for a, _ in b['greedy'])} vs {sum(a for a, _ in f['greedy'])}",
          abs(b["kl"] - f["kl"]) <= 0.01)
if peaks:                                                   # Metal aborts command buffers it cannot fit, and torch
    rec = torch.mps.recommended_max_memory()               # 2.14 then returns garbage silently (src/backend/__init__)
    worst = max(peaks, key=peaks.get)                       # 0.95: where MLX starts freeing its cache
    check(f"GPU memory: every run stays under 0.95 x Metal's recommended {rec / 2**30:.2f} GiB (highest: {worst[0]}, "
          f"{str(worst[1])[6:]} KV, {peaks[worst] / 2**30:.2f} GiB)", all(p <= 0.95 * rec for p in peaks.values()))
ok, what = metal_aborts(started)                            # torch 2.14 never reads a command buffer's status
check(f"Metal aborted no command buffer of this process ({what})", ok)
print(f"      {time.perf_counter() - t0:.0f} s in all")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
