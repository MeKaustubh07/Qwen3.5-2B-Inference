from dataclasses import dataclass

import torch

from ops import softmax

MIN_TEMPERATURE = 1e-5


@dataclass
class SamplingParams:
    temperature: float = 0.7        # <1 sharpens the distribution, >1 flattens it; 0 means greedy
    top_k: int = 20                 # keep only the k most likely tokens (0 = off)
    top_p: float = 0.8              # keep the smallest set whose probabilities add up to p (1.0 = off)
    repetition_penalty: float = 1.0 # >1 discourages tokens that already appeared (1.0 = off; Qwen3.5's setting)
    seed: int | None = None


def sample(logits: torch.Tensor, prev_ids: list[int], params: SamplingParams,
           generator: torch.Generator | None = None) -> int:
    """logits: [vocab] for the next position. Order matches HF: penalty -> temperature -> top-k -> top-p."""
    x = logits.float().clone()
    if params.repetition_penalty != 1.0 and prev_ids:
        seen = torch.tensor(sorted(set(prev_ids)))
        x[seen] = _penalize(x[seen], params.repetition_penalty)
    return _draw(x, None, params, generator)


def _penalize(s: torch.Tensor, penalty: float) -> torch.Tensor:
    return torch.where(s > 0, s / penalty, s * penalty)


def _draw(x: torch.Tensor, ids: torch.Tensor | None, params: SamplingParams,
          generator: torch.Generator | None) -> int:
    """x: penalized logits of the tokens `ids` (ascending; None = the whole vocabulary). Temperature -> top-k ->
    top-p -> one draw. Shared by `sample` and `sample_batch`, so both produce the same token from the same values."""
    if params.temperature < MIN_TEMPERATURE:       # 0 means greedy; so does anything small enough to overflow x / t
        i = int(x.argmax())
        return int(ids[i]) if ids is not None else i
    x = x / params.temperature

    if 0 < params.top_k < x.numel():
        # keep the candidates (ties at the k-th value included, like HF) and work on them only: sorting the full
        # 248k vocabulary for top-p on every token would dominate the step time
        kth = torch.topk(x, params.top_k).values[-1]
        keep = torch.nonzero(x >= kth).flatten()
        ids = keep if ids is None else ids[keep]
        x = x[keep]

    if params.top_p < 1.0:
        order = torch.argsort(x, descending=True)
        probs = softmax(x[order])
        cum = torch.cumsum(probs, dim=0)
        drop = cum - probs >= params.top_p         # drop a token once the tokens ranked above it already cover p
        drop[0] = False                            # always keep the most likely token (HF: min_tokens_to_keep=1)
        x[order[drop]] = float("-inf")

    probs = softmax(x)
    i = int(torch.multinomial(probs, 1, generator=generator))
    return int(ids[i]) if ids is not None else i


CANDIDATE_MARGIN = 8                               # extra candidates beyond top-k, so ties at the k-th value fit
BATCH_TOP_K_MAX = 256                              # rows asking for more take the per-row path, so one request
                                                   # cannot set the candidate count (and cost) for the whole batch


def sample_batch(logits: torch.Tensor, prev_ids: list[list[int]], params: list[SamplingParams],
                 generators: list[torch.Generator | None]) -> list[int | Exception]:
    """One token per row of logits [B, vocab] (any device), identical to calling `sample` row by row.

    The expensive part runs once for the whole batch, on the device the logits are on: the repetition penalty and
    a top-(k + margin) selection by penalized logit (temperature does not change the order). Only those candidates'
    ids and raw logits go on (to the CPU, if the logits are elsewhere), where the penalty is recomputed exactly and
    each row finishes with the same code as `sample`. The server passes CPU logits: on the M2, torch.topk over a
    248k vocabulary is ~4x faster on the CPU than on MPS, and unified memory makes the copy cheap. Rows the candidates cannot represent exactly fall back to `sample`:
    top-k switched off (top-p over the whole vocabulary), top-k above BATCH_TOP_K_MAX, or ties at the k-th value
    that might extend past the candidates.

    A row that cannot be sampled (e.g. NaN logits) yields its exception in place of a token; the other rows are
    unaffected, and each row's generator is drawn from at most once (a seeded request's text never depends on
    which requests it shared a batch with).
    """
    B, V = logits.shape
    x = logits.float()
    penalized = [i for i in range(B) if params[i].repetition_penalty != 1.0 and prev_ids[i]]
    if penalized:
        x = x.clone()
        for i in penalized:
            seen = torch.tensor(sorted(set(prev_ids[i])), device=x.device)
            x[i, seen] = _penalize(x[i, seen], params[i].repetition_penalty)
    def per_row(p: SamplingParams) -> bool:
        return p.temperature >= MIN_TEMPERATURE and not 0 < p.top_k <= min(BATCH_TOP_K_MAX, V - 1)
    ks = [p.top_k for p in params if p.temperature >= MIN_TEMPERATURE and not per_row(p)]
    k = min(V, max(ks, default=1) + CANDIDATE_MARGIN)
    idx = torch.topk(x, k, dim=-1).indices
    raw = torch.gather(logits.float(), 1, idx)
    idx, raw = idx.cpu(), raw.cpu()                              # B x k values (no-op for CPU logits)
    def finish(i: int, p: SamplingParams) -> int:
        if per_row(p):
            return sample(logits[i].float().cpu(), prev_ids[i], p, generators[i])
        order = torch.argsort(idx[i])
        ids, xi = idx[i][order], raw[i][order].clone()
        if i in penalized:
            seen = torch.isin(ids, torch.tensor(sorted(set(prev_ids[i]))))
            xi[seen] = _penalize(xi[seen], p.repetition_penalty)
        if p.temperature >= MIN_TEMPERATURE and torch.sort(xi, descending=True).values[p.top_k - 1] <= xi.min():
            # the k-th value ties with the weakest candidate: its tie group may run past the candidates
            return sample(logits[i].float().cpu(), prev_ids[i], p, generators[i])
        return _draw(xi, ids, p, generators[i])

    out: list[int | Exception] = []
    for i, p in enumerate(params):
        try:
            out.append(finish(i, p))
        except Exception as e:                        # this row alone fails; its batch-mates keep their draws
            out.append(e)
    return out
