"""The machinery shared by decoders whose every layer is attention (Cohere2 / Tiny Aya, Gemma 3 / Sarvam-Translate),
on a pluggable backend. A family supplies block (how attention and the MLP join the residual stream), _project (the
q/k/v projection with its norms and RoPE), _final_norm, and extends embed, head or _qkv where it differs; this class
does the rest for every family:

- weights: each 2-D weight prepared once in the backend's resident format, [q ; k ; v] and [gate ; up] fused;
- state: one KV cache per layer, contiguous or paged (state.GroupedKVPool: the sliding layers keep a ring of blocks per
  sequence, so their memory stops growing past the window), so forks, the pool and the scheduler work unchanged;
- forward, forward_packed, packed_hidden and decode_batch, each refusing a sequence past max_positions (or a chunk the
  ring cannot hold) and reserving its KV blocks before any layer runs, so a refused forward moves nothing.

Sliding window: on the sliding layers the query at position p sees keys max(0, p - W + 1) .. p (transformers:
`kv_idx > q_idx - sliding_window`). They read only that suffix of their KV cache and pass the window to the backend,
which masks the band; the full layers read everything. RoPE positions stay absolute.

stream=True (the fp32 CPU reference of a real model, which would not fit in memory if every widened weight were kept):
each weight is widened when used and dropped after, the embedding rows are read from the stored table, and the tied
head is widened head_rows rows at a time. Slower, the same numbers; peak memory ~1 layer instead of the model.

kv_dtype: what the KV cache stores. fp32 (the default) keeps K/V exactly as computed, as HF does with fp32
activations; bf16 (served on Metal: half the bytes) rounds them as they are stored, and every read widens them back to
fp32, so all arithmetic stays fp32. The rounding is a measured deviation (tests/test_aya_quant.py).
"""
from functools import partial

import torch

from backend.torch_ref import TorchBackend
from models.packing import Segment, advance_all, pack, reserve_all
from quant import concat_rows
from state import ContiguousKVCache, GroupedKVPool, HybridPool, HybridState, WindowedSequence, to_device

HEAD_ROWS = 32768                 # stream=True: rows of the tied head widened at a time (256 MB at hidden 2048)


class AttentionDecoder:
    head_rows = HEAD_ROWS

    def __init__(self, config, weights, backend=None, stream: bool = False, kv_dtype: torch.dtype = torch.float32,
                 prefix: str = "model."):
        if kv_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"kv_dtype must be torch.float32 or torch.bfloat16, not {kv_dtype}")
        self.config = config
        self.kv_dtype = kv_dtype
        self.weights = weights
        self.prefix = prefix
        self.b = backend or TorchBackend()
        self._cache: dict[str, torch.Tensor] = {}
        self.stream = stream
        if stream and (self.b.device.type != "cpu" or getattr(self.b, "weight_dtype", None) != torch.float32):
            raise ValueError("stream=True is the fp32 CPU reference mode")
        names = weights.tensor_names() if hasattr(weights, "tensor_names") else []
        if head := next((n for n in names if n.endswith("lm_head.weight")), None):   # the head is the embedding
            raise ValueError(f"this checkpoint has a separate {head}; {type(self).__name__} ties the head")

    # ---------------------------------------------------------------- weights
    def _get(self, name: str) -> torch.Tensor:
        return self.weights.get(self.prefix + name)

    def _fused(self, key: str, build):
        """A 2-D weight in the backend's resident format (bf16 / quantized / fp32), built by build() once (with
        stream=True on every use, and dropped after)."""
        if self.stream:
            return self.b.prepare(build(), key)
        if key not in self._cache:
            self._cache[key] = self.b.prepare(build(), key)
        return self._cache[key]

    def _w(self, name: str):
        return self._fused(name, lambda: self._get(name))

    def _vec(self, name: str, one_plus: bool = False) -> torch.Tensor:
        """A norm weight, kept in fp32 on the device; one_plus: 1 + w, added in fp32 (Gemma 3's RMSNorm)."""
        key = name + "+1" if one_plus else name
        if key not in self._cache:
            w = self._get(name).float()
            self._cache[key] = (1.0 + w if one_plus else w).to(self.b.device)
        return self._cache[key]

    def _qkv(self, i: int):
        p = f"layers.{i}.self_attn."
        return self._fused(p + "qkv.weight", lambda: concat_rows([self._get(p + f"{n}_proj.weight") for n in "qkv"]))

    def _gate_up(self, i: int):
        p = f"layers.{i}.mlp."
        return self._fused(p + "gate_up.weight",
                           lambda: concat_rows([self._get(p + "gate_proj.weight"), self._get(p + "up_proj.weight")]))

    # ---------------------------------------------------------------- state
    def new_state(self, max_len: int, kv=None) -> HybridState:
        c = self.config
        if kv is None:
            kv = ContiguousKVCache(c.num_hidden_layers, c.num_key_value_heads, c.head_dim, max_len,
                                   device=self.b.device, dtype=self.kv_dtype)
        return HybridState(kv, device=self.b.device)

    def new_paged_pool(self, num_blocks: int, block_size: int = 16, max_seqs: int = 8, max_chunk: int = 512,
                       poison: bool = False) -> HybridPool:
        """num_blocks blocks of every layer, as units of one layer group (state.GroupedKVPool): the sliding layers
        keep a ring of blocks per sequence, so their memory stops growing past the window. max_chunk: the longest
        forward a sequence will run (the scheduler's prefill chunk); a longer one is refused before anything moves."""
        c = self.config
        kv = GroupedKVPool([self._window(i) for i in range(c.num_hidden_layers)], c.num_key_value_heads, c.head_dim,
                           num_blocks, block_size, max_chunk, device=self.b.device, dtype=self.kv_dtype, poison=poison)
        return HybridPool(kv, max_seqs, 0, 0, 0, 0, 0, 1, device=self.b.device)

    def new_paged_state(self, pool: HybridPool) -> HybridState:
        return pool.new_sequence()

    @property
    def max_positions(self) -> int:
        """The longest sequence the model supports: its config's max_position_embeddings."""
        return self.config.max_position_embeddings

    def _check_length(self, end: int) -> None:
        if end > self.max_positions:
            raise ValueError(f"{end} positions: past the model's {self.max_positions}")

    def _window(self, i: int) -> int | None:
        """Layer i's sliding window, or None for a full-attention layer."""
        return self.config.sliding_window if self.config.is_sliding(i) else None

    # ---------------------------------------------------------------- family hooks
    def block(self, i: int, h: torch.Tensor, attend) -> torch.Tensor:
        """Layer i: h [N, hidden] -> h. attend(i, x) is the layer's attention before o_proj, [N, Hq * d]."""
        raise NotImplementedError

    def _project(self, i: int, x: torch.Tensor, positions: torch.Tensor):
        """x [N, hidden] -> q [N, Hq, d], k and v [N, Hkv, d], positional encoding applied."""
        raise NotImplementedError

    def _final_norm(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------------- layers
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        table = self._get("embed_tokens.weight") if self.stream else self._w("embed_tokens.weight")
        return self.b.embedding(table, to_device(ids, self.b.device))

    def _split_qkv(self, y: torch.Tensor):
        """The fused projection [N, (Hq + 2 Hkv) d] -> q [N, Hq, d], k and v [N, Hkv, d]."""
        c = self.config
        Hq, Hkv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        q, k, v = y.split([Hq * d, Hkv * d, Hkv * d], dim=1)
        return q.reshape(-1, Hq, d), k.reshape(-1, Hkv, d), v.reshape(-1, Hkv, d)

    def _rope(self, q, k, positions, theta: float, factor: float = 1.0):
        """RoPE on q and k in one backend call."""
        qk = self.b.rope(torch.cat([q, k], dim=1), positions, theta, factor)
        return qk[:, :q.shape[1]], qk[:, q.shape[1]:]

    def _attend(self, i: int, x, positions, segs: list[Segment]) -> torch.Tensor:
        """x: [N, hidden], the tokens of every segment. Projections run once over all N; attention per segment, over
        its own history."""
        q, k, v = self._project(i, x, positions)
        W, outs = self._window(i), []
        for sg in segs:
            qs, ks, vs = q[sg.lo:sg.hi], k[sg.lo:sg.hi], v[sg.lo:sg.hi]
            if sg.state is not None:
                sg.state.kv.write(i, sg.start, ks, vs)
                lo = max(0, sg.start - W + 1) if W else 0         # the first key the chunk's first query sees
                ks, vs = sg.state.kv.read(i, sg.start + sg.hi - sg.lo, lo)
            outs.append(self.b.attention(qs.contiguous(), ks, vs, causal=True, window=W))   # scale 1/sqrt(d)
        return (torch.cat(outs) if len(outs) > 1 else outs[0]).reshape(x.shape[0], -1)

    def _attend_paged(self, i: int, x, positions, pool: GroupedKVPool, layout) -> torch.Tensor:
        """_attend for a decode step whose sequences share one pool: one KV write and one attention dispatch."""
        blocks, offsets, tables, lens = layout
        q, k, v = self._project(i, x, positions)
        g, slot = pool.layer_map[i]                    # layer i is slot `slot` of group g's units
        pool.write_batch(slot, blocks[g], offsets, k, v)
        o = self.b.paged_attention(q.contiguous(), pool.k[slot], pool.v[slot], tables[g], lens, pool.block_size,
                                   window=self._window(i))
        return o.reshape(x.shape[0], -1)

    def _layers(self, ids, positions, segs: list[Segment], capture: dict | None = None, decode: bool = False):
        """Embedding and all layers over the packed tokens -> hidden states [N, hidden] (before the final norm).
        decode: one token per segment; when their states share one GroupedKVPool, each layer writes and attends for
        all of them at once."""
        for sg in segs:
            self._check_length(sg.start + sg.hi - sg.lo)
            if sg.state is not None and hasattr(sg.state.kv, "check_forward"):   # the ring can hold the chunk
                sg.state.kv.check_forward(sg.start, sg.start + sg.hi - sg.lo)
        reserve_all(segs)                            # fail before any layer mutates state (atomic forward)
        kvs = [sg.state.kv for sg in segs] if decode else []
        if kvs and all(isinstance(kv, WindowedSequence) and kv.pool is kvs[0].pool for kv in kvs):
            pool = kvs[0].pool
            attend = partial(self._attend_paged, positions=positions, pool=pool,
                             layout=pool.batch_layout(kvs, [sg.start for sg in segs]))
        else:
            attend = partial(self._attend, positions=positions, segs=segs)
        h = self.embed(ids)
        if capture is not None:
            capture["embed"] = h
        for i in range(self.config.num_hidden_layers):
            h = self.block(i, h, attend)
            if capture is not None:
                capture[f"l{i}_out"] = h
        advance_all(segs)
        return h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden states [N, hidden] -> logits [N, vocab]: the tied head."""
        if not self.stream:
            return self.b.linear(h, self._w("embed_tokens.weight"))
        table, n = self._get("embed_tokens.weight"), self.head_rows
        return torch.cat([self.b.linear(h, self.b.prepare(table[r:r + n])) for r in range(0, table.shape[0], n)], dim=1)

    def forward(self, ids: torch.Tensor, state: HybridState | None = None, last_only: bool = False,
                capture: dict | None = None) -> torch.Tensor:
        """Token ids [T] -> logits [T, vocab]. With a state, continues from state.length and updates it."""
        start = state.length if state is not None else 0
        positions = torch.arange(start, start + len(ids), device=self.b.device, dtype=torch.int32)
        h = self._layers(ids, positions, [Segment(0, len(ids), state, start)], capture)
        h = self._final_norm(h[-1:] if last_only else h)
        if capture is not None:
            capture["final_norm"] = h
        return self.head(h)

    def forward_packed(self, chunks: list[tuple[torch.Tensor, HybridState]]) -> torch.Tensor:
        """Several sequences' chunks (each continuing its own state) in one pass -> logits of each chunk's last
        token [len(chunks), vocab]. Same result as calling forward on each chunk, with each weight read once."""
        ids, positions, segs = pack(chunks, self.b.device)
        h = self._layers(ids, positions, segs)
        last = to_device(torch.tensor([sg.hi - 1 for sg in segs]), self.b.device)
        return self.head(self._final_norm(h[last]))

    def packed_hidden(self, chunks: list[tuple[torch.Tensor, HybridState]],
                      capture: dict | None = None) -> torch.Tensor:
        """Like forward_packed, but -> the final-norm hidden state of EVERY token [N, hidden], rows in chunk order.
        capture gets the packed embedding, every layer's output and the final norm (rows in chunk order too)."""
        ids, positions, segs = pack(chunks, self.b.device)
        h = self._final_norm(self._layers(ids, positions, segs, capture))
        if capture is not None:
            capture["final_norm"] = h
        return h

    def decode_batch(self, tokens: list[int], states: list) -> torch.Tensor:
        """One decode step for B independent sequences -> logits [B, vocab]. Projections and MLPs run batched (each
        weight read once per step for everyone); attention reads each sequence's own KV history."""
        segs = [Segment(j, j + 1, st, st.length) for j, st in enumerate(states)]
        positions = to_device(torch.tensor([sg.start for sg in segs], dtype=torch.int32), self.b.device)
        return self.head(self._final_norm(self._layers(torch.tensor(tokens), positions, segs, decode=True)))
