from typing import NamedTuple

import torch

from state import to_device


class Segment(NamedTuple):
    lo: int              # this sequence's rows in the packed batch: [lo, hi)
    hi: int
    state: object        # its KV cache / hybrid state, or None (stateless forward)
    start: int           # position of its first token in this pack (= tokens already in its state)


def pack(chunks: list[tuple[torch.Tensor, object]], device) -> tuple[torch.Tensor, torch.Tensor, list[Segment]]:
    """chunks: [(ids [T_i], state_i)] -> (concatenated ids, per-token positions on `device`, segments).
    Each chunk continues its own state from state.length (a later chunk of a long prompt, or a whole prompt)."""
    segs, pos, lo = [], [], 0
    for ids, st in chunks:
        start = st.length if st is not None else 0
        segs.append(Segment(lo, lo + len(ids), st, start))
        pos.append(torch.arange(start, start + len(ids), dtype=torch.int32))
        lo += len(ids)
    ids = torch.cat([c[0] for c in chunks])
    return ids, to_device(torch.cat(pos), device), segs


def reserve_all(segs: list[Segment]) -> None:
    """Allocate every segment's KV blocks before any layer runs, so running out fails without touching state."""
    for sg in segs:
        if sg.state is not None:
            sg.state.reserve(sg.start + sg.hi - sg.lo)


def advance_all(segs: list[Segment]) -> None:
    for sg in segs:
        if sg.state is not None:
            sg.state.advance(sg.hi - sg.lo)
