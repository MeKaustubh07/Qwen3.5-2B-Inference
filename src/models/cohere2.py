"""Cohere2 (Tiny Aya): every layer is attention, on a pluggable backend.

Follows the verified spec of HF transformers' modeling_cohere2.py (docs/tiny-aya-plan.md). What differs from Qwen3.5:
- no DeltaNet: all 36 layers are attention. Three sliding-window layers (with RoPE) are followed by one full layer
  with no positional encoding at all, repeating;
- one LayerNorm per layer (mean subtracted, plain weight, no bias) feeds attention AND the MLP, in parallel:
  h + attn(LN(h)) + mlp(LN(h));
- RoPE rotates neighbouring numbers (2i, 2i+1) ("interleaved", GPT-J) instead of (i, i + d/2), at theta 50000, over
  all head dims;
- logits are multiplied by logit_scale; no QK-norm, no output gate, no biases; the head is tied to the embedding.

Interleaved RoPE without a new kernel: within each head, the rows of W_q and W_k are reordered at load time so that
numbers 2i and 2i+1 land at i and i + d/2. The half-split rotation then turns exactly the pairs HF turns, at the
same frequencies, and q.k does not change when q and k get the same reordering (cached keys stay reordered).

The machinery every all-attention family shares (sliding window, KV state, forward paths, stream=True, kv_dtype)
is models/decoder.py's.
"""
import torch

from models.decoder import AttentionDecoder
from quant import concat_rows, select_rows


def interleaved_to_half(n_heads: int, d: int) -> torch.Tensor:
    """Row order that moves each head's RoPE pair (2i, 2i+1) to (i, i + d/2): evens first, then odds, per head."""
    per_head = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    return (torch.arange(n_heads)[:, None] * d + per_head[None, :]).flatten()


class Cohere2Model(AttentionDecoder):
    def _qkv(self, i: int):
        """Fused [q ; k ; v] projection; on RoPE (sliding) layers, q and k rows reordered for interleaved RoPE."""
        c, p = self.config, f"layers.{i}.self_attn."

        def build():
            q, k, v = (self._get(p + f"{n}_proj.weight") for n in "qkv")
            if c.is_sliding(i):
                q = select_rows(q, interleaved_to_half(c.num_attention_heads, c.head_dim))
                k = select_rows(k, interleaved_to_half(c.num_key_value_heads, c.head_dim))
            return concat_rows([q, k, v])
        return self._fused(p + "qkv.weight", build)

    def _project(self, i: int, x, positions):
        """RoPE on sliding layers only: the full layers have no positional encoding."""
        q, k, v = self._split_qkv(self.b.linear(x, self._qkv(i)))
        if self.config.is_sliding(i):
            q, k = self._rope(q, k, positions, self.config.rope_theta)
        return q, k, v

    def block(self, i: int, h, attend):
        b, p = self.b, f"layers.{i}."
        x = b.layer_norm(h, self._vec(p + "input_layernorm.weight"), self.config.layer_norm_eps)
        a = b.linear(attend(i, x), self._w(p + "self_attn.o_proj.weight"), residual=h)          # h + attention
        return b.linear(b.swiglu(x, self._gate_up(i)), self._w(p + "mlp.down_proj.weight"), residual=a)   # + mlp

    def _final_norm(self, h):
        return self.b.layer_norm(h, self._vec("norm.weight"), self.config.layer_norm_eps)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden states [N, hidden] -> logits [N, vocab]: the tied head, times logit_scale."""
        logits = super().head(h)
        return logits * self.config.logit_scale if self.config.logit_scale != 1.0 else logits
