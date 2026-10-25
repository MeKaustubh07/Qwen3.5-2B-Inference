import torch

from config import Gemma3Config
from models.decoder import AttentionDecoder

PREFIXES = ("language_model.model.", "model.")


def text_prefix(weights) -> str:
    """The prefix of the text model's tensors in this checkpoint."""
    names = set(weights.tensor_names())
    for p in PREFIXES:
        if p + "embed_tokens.weight" in names:
            return p
    raise ValueError(f"no Gemma 3 text model in this checkpoint (no embed_tokens.weight under {' or '.join(PREFIXES)})")


class Gemma3Model(AttentionDecoder):
    def __init__(self, config: Gemma3Config, weights, backend=None, stream: bool = False,
                 kv_dtype: torch.dtype = torch.float32, embed_scale_dtype: torch.dtype = torch.float32):
        if not embed_scale_dtype.is_floating_point:
            raise ValueError(f"embed_scale_dtype must be a floating-point dtype, not {embed_scale_dtype}")
        super().__init__(config, weights, backend, stream, kv_dtype, prefix=text_prefix(weights))
        self.embed_scale = torch.tensor(config.hidden_size ** 0.5, dtype=torch.float32).to(embed_scale_dtype).item()

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return self.b.rms_norm(x, self._vec(name, one_plus=True), self.config.rms_norm_eps)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return super().embed(ids) * self.embed_scale

    def _project(self, i: int, x, positions):
        """q and k RMS-normed per head, then RoPE with the layer type's theta and linear factor."""
        p = f"layers.{i}.self_attn."
        q, k, v = self._split_qkv(self.b.linear(x, self._qkv(i)))
        q, k = self._rope(self._norm(q, p + "q_norm.weight"), self._norm(k, p + "k_norm.weight"), positions,
                          *self.config.rope_for(i))
        return q, k, v

    def block(self, i: int, h, attend):
        b, p = self.b, f"layers.{i}."
        a = b.linear(attend(i, self._norm(h, p + "input_layernorm.weight")), self._w(p + "self_attn.o_proj.weight"))
        h = h + self._norm(a, p + "post_attention_layernorm.weight")
        m = b.linear(b.geglu(self._norm(h, p + "pre_feedforward_layernorm.weight"), self._gate_up(i)),
                     self._w(p + "mlp.down_proj.weight"))
        return h + self._norm(m, p + "post_feedforward_layernorm.weight")

    def _final_norm(self, h):
        return self._norm(h, "norm.weight")
