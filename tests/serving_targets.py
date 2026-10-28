import argparse
import gc
import os
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "tests")
from engine import Engine, load_engine


@dataclass
class Target:
    name: str
    engine: Engine
    exact: bool                       # fp32 on the CPU: batched / packed / chunked runs equal sequential ones
    real: bool                        # a trained model: checks of meaning (Paris, 7 + 3, review scores) apply
    thinking: bool                    # the model has a <think> mode
    logit_tol: float                  # batched vs sequential logits
    logprob_tol: float                # decision scores vs references
    prefill_chunk: int = 512          # what server apps built by the suites use
    step_delay: float = 0.0           # seconds added to every decode step, for tests that need a stream in flight
    keep: list = field(default_factory=list)   # objects that must outlive the target (its temp directory)
    started: str = ""                 # local time just before the model loaded: Metal aborts are counted from it
    gate_meaning: bool = True         # checks of meaning gate; False: reported only (a quantized model, D10)

    @property
    def gpu(self) -> bool:
        return self.engine.model.b.device.type == "mps"

    @property
    def tie(self) -> float:
        """A served token other than the reference's pick is a near-tie when the reference ranks it less than this
        below its pick: two runs whose logits differ by at most logit_tol can order such a pair either way (D9)."""
        return 2 * self.logit_tol

    def agree(self, ref, got: list[int], finish: str | None = None, what: str = "") -> bool:
        """Served greedy ids `got` == a sequential reference ref = (tokens, logits rows), which has one more row than
        tokens when it stopped on an end token, for the same max_tokens. Exact targets: equal ids. The others (D9):
        equal ids (a cancelled request may stop short of the reference), or a near-tie at the first difference, where
        the reference ranks the served token (an end token if the request stopped there: finish 'stop') less than
        `tie` below its own pick; later tokens follow different contexts and are not compared. Each near-tie,
        divergence or wrong length is noted on the next check's label (notes())."""
        toks, logs = ref
        if self.exact:
            return list(got) == list(toks)
        end = object()
        r_seq = list(toks) + ([end] if len(logs) > len(toks) else [])
        s_seq = list(got) + ([end] if finish == "stop" else [])
        k = next((i for i, (a, b) in enumerate(zip(r_seq, s_seq)) if a != b), None)
        if k is None:                              # one is a prefix of the other: only a cancel may stop short
            if len(s_seq) == len(r_seq) or finish == "cancelled" and len(s_seq) < len(r_seq):
                return True
            _notes.append(f"; {what + ': ' if what else ''}{len(got)} tokens ({finish}), the reference {len(toks)}")
            return False
        row, eos = logs[k], sorted(self.engine.eos_ids)
        def logit(t):
            return row[eos].max().item() if t is end else row[t].item()
        gap = logit(r_seq[k]) - logit(s_seq[k])
        _notes.append(f"; {what + ': ' if what else ''}{'near-tie' if gap < self.tie else 'diverged'} at token {k} "
                      f"({gap:.1e} below the reference's pick, {'<' if gap < self.tie else '>='} {self.tie:.0e})")
        return gap < self.tie

    def tidy(self) -> None:
        """Between sub-suites on a GPU target: free the apps, schedulers and pools they built (memory is gated)."""
        if self.gpu:
            release_apps()
            gc.collect()
            torch.mps.empty_cache()

    def raw_ids(self, text: str) -> list[int]:
        """A raw prompt's ids as /v1/completions builds them."""
        return self.engine.tokenizer.encode(text)

    def chat_len(self) -> int:
        """Tokens of a one-line chat: suites size their max_model_len from it."""
        from chat import format_chat
        return len(self.engine.tokenizer.encode(format_chat([{"role": "user", "content": "Hi"}],
                                                            style=self.engine.chat_style)))

    def slow_steps(self, sched) -> None:
        """Make each decode step take at least step_delay, so a stream is still running when a test acts on it."""
        if self.step_delay:
            real = sched._decode_step
            def step(*a, **k):
                time.sleep(self.step_delay)
                return real(*a, **k)
            sched._decode_step = step


def na(name: str) -> None:
    print(f"N/A   {name}")


_notes: list[str] = []


def notes() -> str:
    """The notes of the comparisons since the last call (near-ties, divergences), for a check's label. Empty on exact
    targets, so their output is unchanged."""
    s = "".join(_notes)
    _notes.clear()
    return s


def report(name: str, ok: bool) -> None:
    """A check of meaning on a quantized model (D10): printed, not gated (run_tests.py counts PASS, FAIL, SKIP and the
    not-applicable mark anywhere in a line, so a report says neither)."""
    print(f"INFO  {name}: {'holds' if ok else 'does not hold'} (reported, not gated)")


def agree_selfcheck() -> bool:
    """The near-tie rule on made-up logits over 5 ids, end token 3: what it accepts and what it refuses."""
    from types import SimpleNamespace
    t = Target("made-up", SimpleNamespace(eos_ids={3}), exact=False, real=False, thinking=False, logit_tol=1e-2,
               logprob_tol=0.0)
    def rows(*steps):                              # per step: (the pick, a runner-up, how far below the pick it is)
        out = []
        for top, second, gap in steps:
            r = torch.zeros(5)
            r[top], r[second] = 1.0, 1.0 - gap
            out.append(r)
        return out
    ref = ([1, 2], rows((1, 4, 0.5), (2, 4, 0.01)))
    stops = ([1], rows((1, 4, 0.5), (3, 2, 0.01)))              # the reference ends at step 1 (id 2 close behind)
    ends_close = ([1, 2], rows((1, 3, 0.01), (2, 4, 0.5)))      # the end token 0.01 below at step 0
    ok = [t.agree(ref, [1, 2]), t.agree(ref, [1], "cancelled"), t.agree(ref, [1, 4]), not t.agree(ref, [4, 2]),
          not t.agree(ref, [1, 0]), t.agree(stops, [1], "stop"), t.agree(stops, [1, 2], "length"),
          not t.agree(stops, [1, 4], "length"), t.agree(ends_close, [], "stop"), not t.agree(ref, [], "stop"),
          not t.agree(ref, [1], "length"), not t.agree(ref, [1, 2, 0], "length"),
          not t.agree(ref, [1, 2, 0], "cancelled"),
          not Target("exact", t.engine, True, False, False, 1e-2, 0.0).agree(ref, [1, 4])]
    _notes.clear()
    return all(ok)


class Served:
    """Records the Requests a scheduler is given, so a check can read the ids it generated (HTTP returns text)."""

    def __init__(self, sched):
        self.reqs, submit = [], sched.submit
        def recording(ids, *a, **k):
            req = submit(ids, *a, **k)
            self.reqs.append(req)
            return req
        sched.submit = recording

    def last(self, ids: list[int]):
        """The latest request with these prompt ids."""
        return next(r for r in reversed(self.reqs) if r.prompt_ids == list(ids))


def release_apps() -> None:
    """Drop FastAPI's references to every endpoint create_app made: module-level lru_caches in
    fastapi.dependencies.models keep the endpoint functions, and those hold their scheduler, its pool and the engine."""
    try:
        import fastapi.dependencies.models as fm
    except ImportError:
        return
    for f in vars(fm).values():
        if callable(getattr(f, "cache_clear", None)):
            f.cache_clear()


def gpu_peak() -> int:
    """This process's GPU memory high point: the allocator's peak heap bytes, plus what Metal holds outside them
    now."""
    torch.mps.synchronize()
    A = torch.accelerator
    return A.max_memory_reserved() + max(0, torch.mps.driver_allocated_memory() - A.memory_reserved())


ABORT = "Execution of the command buffer was aborted"


def metal_aborts(since: str) -> tuple[bool, str]:
    """Whether Metal aborted none of this process's command buffers since `since` (local time), read from macOS's
    log: PyTorch 2.14 never reads a command buffer's status, so an out-of-memory abort is otherwise silent.
    -> (none aborted, what the log said)"""
    try:
        log = subprocess.run(["/usr/bin/log", "show", "--start", since, "--style", "compact", "--predicate",
                              f'processID == {os.getpid()} AND eventMessage CONTAINS "{ABORT}"'],
                             capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "macOS's log not readable: log show timed out after 300 s"
    if log.returncode != 0:                                  # it needs an admin account and no sandbox
        return False, f"macOS's log not readable (log show: {log.returncode}, {log.stderr.strip()[:150]})"
    aborts = [line.split(ABORT)[-1].strip() for line in log.stdout.splitlines() if ABORT in line]
    return not aborts, f"macOS log: {len(aborts)}" + (f", e.g. {aborts[0][:90]}" if aborts else "")


def gpu_gates(t: Target, check) -> None:
    """On a GPU target, two gates over the whole suite: peak GPU memory at most 0.95 x what Metal recommends (where
    MLX starts freeing its cache), and no command buffer aborted by Metal. Nothing on a CPU target. Call it while the
    target is loaded."""
    if not t.gpu:
        return
    peak, rec = gpu_peak(), torch.mps.recommended_max_memory()
    check(f"GPU memory: peak {peak / 2**30:.2f} GiB <= 0.95 x the {rec / 2**30:.2f} GiB Metal recommends",
          peak <= 0.95 * rec)
    ok, what = metal_aborts(t.started)
    check(f"Metal aborted no command buffer of this process ({what})", ok)


PROMPTS = ["The capital of France is", "def fibonacci(n):",
           "The history of computing is a story of abstraction. Each generation of engineers built tools that",
           "Hello world"]


_cache: dict[str, Target] = {}
_lock = threading.Lock()


def load_target(name: str) -> Target:
    with _lock:
        if name in _cache:
            return _cache[name]
        if name == "qwen":
            t = Target("qwen3.5-0.8b", load_engine("qwen3.5-0.8b", "cpu"), exact=True, real=True, thinking=True,
                       logit_tol=2e-3, logprob_tol=2e-4)
        else:
            raise ValueError(f"unknown target {name!r} (qwen)")
        _cache[name] = t
        return t


def unload_target(name: str) -> bool:
    """Forget a loaded target and free its model, before a suite loads Qwen on the GPU (the fp32 CPU Qwen is 3.2 GB of
    an 8 GB machine). The caller drops its own references first (the engine, its schedulers). FastAPI keeps every
    endpoint function create_app made in module-level lru_caches (fastapi.dependencies.models), and those functions
    hold the engine, so the caches are cleared here. -> whether the model is gone."""
    with _lock:
        t = _cache.pop(name, None)
    if t is None:
        return True
    model, gpu = weakref.ref(t.engine.model), t.gpu
    del t
    release_apps()
    gc.collect()
    if gpu:
        torch.mps.empty_cache()
    return model() is None


def target_from_argv(default: str = "qwen") -> str:
    """--target qwen (or --target=qwen); anything else on the command line is an error."""
    p = argparse.ArgumentParser(description="a server suite (tests/serving_targets.py)")
    p.add_argument("--target", choices=("qwen",), default=default)
    return p.parse_args().target

