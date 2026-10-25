import torch

from chat import TemplateChat, format_chat
from state import GroupedKVPool

PROBES = ("Hi", "What is the capital of France?")   # two one-line chats: what they share is the preamble


class PinnedPrefix:
    def __init__(self, model, kv: GroupedKVPool, ids: list[int], chunk: int):
        self.ids, self.n, bs = ids, len(ids) // kv.block_size, kv.block_size
        self.cap = kv.ring * bs                        # prompt + max_tokens of a hit: no ring ever wraps onto it
        self.state = model.new_state(0, kv=kv.new_sequence())   # holds no batch slot (HybridPool.new_sequence would)
        for i in range(0, len(ids), chunk):            # chunks the pool's rings were sized for
            model.forward_packed([(torch.tensor(ids[i:i + chunk]), self.state)])
        self.units = sum(map(len, self.state.kv.tables))

    @classmethod
    def build(cls, sched) -> "tuple[PinnedPrefix | None, str]":
        """The model's preamble pinned in the scheduler's pool, or None and why not. Runs before the engine thread
        starts (the constructing thread still owns the GPU)."""
        kv, style = getattr(sched.pool, "kv", None), getattr(sched.eng, "chat_style", None)
        if not isinstance(kv, GroupedKVPool) or kv.ring is None or not isinstance(style, TemplateChat):
            return None, "needs a windowed KV pool and the model's own chat template"
        enc = [sched.tok.encode(format_chat([{"role": "user", "content": t}], style=style), add_bos=False)
               for t in PROBES]                        # as /v1/chat/completions encodes a chat
        common = next((i for i, (a, b) in enumerate(zip(*enc)) if a != b), min(map(len, enc)))
        n = common // kv.block_size
        if n == 0 or n >= kv.ring or n * kv.block_size >= min(map(len, enc)):
            return None, f"a {common}-token common prefix does not fit whole blocks inside the {kv.ring}-block ring"
        if kv.blocks_for(n * kv.block_size) > kv.allocator.num_free:
            return None, "the KV pool cannot hold it"
        return cls(sched.model, kv, enc[0][:n * kv.block_size], sched.prefill_chunk), "ok"

    def match(self, ids: list[int], max_new: int) -> tuple[int, bool]:
        """-> (tokens served from the preamble, 0 on a miss; whether it matched but was too long: a bypass)."""
        starts = len(ids) > len(self.ids) and ids[:len(self.ids)] == self.ids
        hit = starts and len(ids) + max_new <= self.cap
        return (len(self.ids) if hit else 0), starts and not hit

    def attach(self, state) -> None:
        state.kv.borrow(self.state.kv, self.n)
