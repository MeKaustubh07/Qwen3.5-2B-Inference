import json
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, "src")
from engine import MODELS

D = "models/sarvam-translate"
BUILDS = {"metal-int8": "model.int8.qt", "metal-int4": "model.int4.qt"}
if not os.path.exists(f"{D}/model.safetensors.index.json") or not torch.backends.mps.is_available():
    print(f"SKIP: needs the checkpoint in {D} and a Metal GPU")
    sys.exit(0)

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


t0, rec = time.perf_counter(), torch.mps.recommended_max_memory()
spec = MODELS["sarvam-translate"]
for backend, file in BUILDS.items():
    qt = f"{D}/{file}"
    if not os.path.exists(qt):
        check(f"{backend}: {qt} exists (scripts/quantize.py; see tests/test_gemma3_quant.py)", False)
        continue
    budget = spec | spec["per_backend"].get(backend, {})
    with tempfile.TemporaryDirectory() as tmp:          # one process per build: the server holds one model
        out = f"{tmp}/mix.jsonl"
        run = subprocess.run([sys.executable, "scripts/bench.py", "--model", "sarvam-translate", "--backend", backend,
                              "--weights", qt, "--serve-mix", "--out", out], capture_output=True, text=True)
        rows = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
    print("\n".join(f"      {line}" for line in run.stdout.strip().splitlines()), flush=True)
    end = next((r for r in rows if r["phase"] == "end"), None)
    if run.returncode != 0 or end is None:
        check(f"{backend}: bench.py --serve-mix ran (exit {run.returncode}: {run.stderr.strip()[-300:]})", False)
        continue
    runs = [r for r in rows if r["phase"] != "end"]
    worst = max(runs, key=lambda r: r["peak"])
    used = (runs[0]["kv_blocks"], runs[0]["max_len"], runs[0]["decide_len"])
    want = (budget["kv_blocks"], budget["max_model_len"], budget.get("decide_max_len", budget["max_model_len"]))
    check(f"{backend}: the registry's budgets measured (pool {used[0]} blocks, generation {used[1]} tokens, "
          f"/v1/decide {used[2]}) in one process: {len({r['phase'] for r in runs})} workloads x "
          f"{max(r['run'] for r in runs)} runs, each decide also interleaved with generation",
          used == want and any("interleaved" in r["phase"] for r in runs))
    check(f"{backend}: every run's GPU peak stays under 0.95 x Metal's recommended {rec / 2**30:.2f} GiB (highest: "
          f"{worst['phase']}, run {worst['run']}, {worst['peak'] / 2**30:.2f} GiB = {worst['ratio']:.3f}x)",
          worst["peak"] <= 0.95 * rec)
    check(f"{backend}: Metal aborted no command buffer of the run (macOS log: {end['aborts']})", end["aborts"] == 0)
print(f"      {time.perf_counter() - t0:.0f} s in all")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
