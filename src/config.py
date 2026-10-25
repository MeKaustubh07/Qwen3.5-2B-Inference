import copy
import json
from dataclasses import dataclass

LAYER_TYPES = {"sliding_attention", "full_attention"}


def _positive_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _positive_real(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _check_attention_layers(layer_types, n_layers: int, window, bad: dict) -> None:
    """One type per layer, each sliding or full, and a positive window: the layouts the engine implements."""
    if not isinstance(layer_types, list) or len(layer_types) != n_layers or set(layer_types) - LAYER_TYPES:
        bad["layer_types"] = layer_types
    if not _positive_int(window):
        bad["sliding_window"] = window


@dataclass
class Qwen35Config:
    """Qwen3.5 text model (hybrid Gated DeltaNet + gated full attention). Fields live under config["text_config"]."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int          # 8 query heads (each with a same-size output gate)
    num_key_value_heads: int          # 2
    head_dim: int                     # 256, explicit (NOT hidden_size // heads)
    rms_norm_eps: float
    rope_theta: float                 # 1e7
    partial_rotary_factor: float      # 0.25 -> only the first 64 of 256 dims rotate
    layer_types: list[str]            # "linear_attention" | "full_attention", one per layer
    linear_conv_kernel_dim: int       # 4
    linear_key_head_dim: int          # 128
    linear_value_head_dim: int        # 128
    linear_num_key_heads: int         # 16
    linear_num_value_heads: int       # 16
    tie_word_embeddings: bool

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_json(cls, path: str) -> "Qwen35Config":
        top = json.load(open(path))
        c = top["text_config"]
        rope = c["rope_parameters"]
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"], rms_norm_eps=c["rms_norm_eps"],
            rope_theta=float(rope["rope_theta"]), partial_rotary_factor=rope["partial_rotary_factor"],
            layer_types=list(c["layer_types"]), linear_conv_kernel_dim=c["linear_conv_kernel_dim"],
            linear_key_head_dim=c["linear_key_head_dim"], linear_value_head_dim=c["linear_value_head_dim"],
            linear_num_key_heads=c["linear_num_key_heads"], linear_num_value_heads=c["linear_num_value_heads"],
            tie_word_embeddings=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
        )


@dataclass
class Cohere2Config:
    """Cohere2 (Tiny Aya): every layer is attention. Sliding-window layers (RoPE) alternate with full layers (no
    positional encoding); one LayerNorm per layer feeds attention and the MLP in parallel; the embedding is tied."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int          # 16 query heads
    num_key_value_heads: int          # 4
    head_dim: int                     # 128 = hidden_size // heads
    layer_norm_eps: float
    rope_theta: float                 # 50000, interleaved pairs, all head_dim dims, sliding layers only
    sliding_window: int               # 4096: a token sees itself and the 4095 before it
    layer_types: list[str]            # "sliding_attention" | "full_attention", one per layer
    logit_scale: float                # logits are multiplied by this (1.0 for Tiny Aya)
    tie_word_embeddings: bool
    max_position_embeddings: int

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    @classmethod
    def from_json(cls, path: str) -> "Cohere2Config":
        c = json.load(open(path))
        # The settings this engine implements. Some change transformers' graph (attention_bias, hidden_act,
        # rope_scaling); the original-Cohere keys (use_parallel_block, rotary_pct, ...) are ignored by transformers,
        # which always builds the Tiny Aya graph. Either way a config asking for anything else is refused here,
        # instead of producing wrong text.
        want = {"model_type": "cohere2", "use_parallel_block": True, "use_qk_norm": False, "attention_bias": False,
                "rotary_pct": 1.0, "position_embedding_type": "rope_gptj", "rope_scaling": None, "hidden_act": "silu",
                "use_gated_activation": True, "use_parallel_embedding": False, "use_embedding_sharing": True}
        bad = {k: c.get(k) for k, v in want.items() if k in c and c[k] != v}
        if c.get("model_type") != "cohere2":
            bad["model_type"] = c.get("model_type")
        heads = c["num_attention_heads"]
        head_dim = c["hidden_size"] // heads              # transformers takes config.json's head_dim when present;
        if c.get("head_dim", head_dim) != head_dim:       # this engine supports only hidden_size // heads
            bad["head_dim"] = c["head_dim"]
        layer_types = list(c["layer_types"])
        _check_attention_layers(layer_types, c["num_hidden_layers"], c.get("sliding_window"), bad)
        if c.get("tie_word_embeddings", True) is not True:  # the head is the embedding: an untied lm_head
            bad["tie_word_embeddings"] = c["tie_word_embeddings"]     # would be ignored, so refuse it
        # transformers 5 nests rope_theta under rope_parameters (older files keep it at the top level), and reads the
        # scaling type from "rope_type", or else the legacy "type"
        rope = c.get("rope_parameters") or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default" or set(rope) - {"rope_type", "type", "rope_theta"}:
            bad["rope_parameters"] = rope
        if bad:
            raise ValueError(f"unsupported Cohere2 settings: {bad}")
        rope_theta = rope.get("rope_theta", c.get("rope_theta"))
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=heads,
            num_key_value_heads=c["num_key_value_heads"], head_dim=head_dim, layer_norm_eps=c["layer_norm_eps"],
            rope_theta=float(rope_theta), sliding_window=c["sliding_window"], layer_types=layer_types,
            logit_scale=float(c.get("logit_scale", 0.0625)),          # transformers' default when the key is absent
            tie_word_embeddings=c.get("tie_word_embeddings", True),   # Tiny Aya's file never says; the default ties
            max_position_embeddings=c["max_position_embeddings"],
        )


# Gemma3TextConfig's defaults (transformers configuration_gemma3.py:77-101): what a key the file leaves out means
_GEMMA3_TEXT_DEFAULTS = dict(
    model_type="gemma3_text", vocab_size=262_208, hidden_size=2304, intermediate_size=9216, num_hidden_layers=26,
    num_attention_heads=8, num_key_value_heads=4, head_dim=256, hidden_activation="gelu_pytorch_tanh",
    max_position_embeddings=131_072, rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=False,
    query_pre_attn_scalar=256, sliding_window=4096, final_logit_softcapping=None, attn_logit_softcapping=None,
    use_bidirectional_attention=False)


@dataclass
class Gemma3Config:
    """Gemma 3 text model (Sarvam-Translate): every layer is attention, five sliding-window layers to one full layer,
    each type with its own RoPE; RMSNorm (1 + w) before and after attention and the MLP, q/k norms, GeGLU, a scaled
    tied embedding. Read from a Gemma3ForConditionalGeneration config (the text model under text_config, legacy RoPE
    keys, layer types from sliding_window_pattern) or a gemma3_text one (rope_parameters per layer type,
    layer_types)."""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int          # 8 query heads
    num_key_value_heads: int          # 4
    head_dim: int                     # 256, explicit (8 x 256 != hidden 2560)
    rms_norm_eps: float
    sliding_window: int               # 1024: a token sees itself and the 1023 before it
    layer_types: list[str]            # "sliding_attention" | "full_attention", one per layer
    rope: dict[str, tuple[float, float]]   # layer type -> (theta, linear factor): sliding (1e4, 1), full (1e6, 8)
    tie_word_embeddings: bool
    max_position_embeddings: int      # Gemma 3's 131072; the registry caps Sarvam-Translate at 8192

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def rope_for(self, layer: int) -> tuple[float, float]:
        return self.rope[self.layer_types[layer]]

    @classmethod
    def from_json(cls, path: str) -> "Gemma3Config":
        top = json.load(open(path))
        if top.get("model_type") == "gemma3" and isinstance(top.get("text_config"), dict):
            text = top["text_config"]
        elif top.get("model_type") == "gemma3_text":
            text = top
        else:
            raise ValueError(f"not a Gemma 3 text config: model_type {top.get('model_type')!r}")
        c = _GEMMA3_TEXT_DEFAULTS | text
        allowed = {"model_type": ("gemma3_text",), "hidden_activation": ("gelu_pytorch_tanh",),
                   "attention_bias": (False,), "final_logit_softcapping": (None,), "attn_logit_softcapping": (None,),
                   "use_bidirectional_attention": (False, None), "tie_word_embeddings": (True,)}
        bad = {k: c[k] for k, ok in allowed.items() if c[k] not in ok}
        if top.get("tie_word_embeddings", True) is not True:
            bad["tie_word_embeddings"] = top["tie_word_embeddings"]
        if not _positive_int(c["head_dim"]):
            bad["head_dim"] = c["head_dim"]
        if c["query_pre_attn_scalar"] != c["head_dim"]:        # the attention op scales scores by 1/sqrt(head_dim),
            bad["query_pre_attn_scalar"] = c["query_pre_attn_scalar"]   # transformers by query_pre_attn_scalar^-0.5
        n, layer_types = c["num_hidden_layers"], c.get("layer_types")
        if layer_types is None:                                 # configuration_gemma3.py:109-113
            pattern = c.get("sliding_window_pattern", 6)
            if _positive_int(pattern):
                layer_types = ["sliding_attention" if (i + 1) % pattern else "full_attention" for i in range(n)]
            else:
                bad["sliding_window_pattern"] = pattern
        _check_attention_layers(layer_types, n, c["sliding_window"], bad)
        # convert_rope_params_to_dict (configuration_gemma3.py:136-163): the legacy keys only fill what
        # rope_parameters leaves out, and rope_scaling is merged into the full layers' entry, so a legacy
        # {"type": "linear"} next to the default rope_type "default" stays default (standardize_rope_params)
        rope = copy.deepcopy(c.get("rope_parameters")) or {}
        if not isinstance(rope, dict) or set(rope) - LAYER_TYPES \
                or any(v is not None and not isinstance(v, dict) for v in rope.values()):
            bad["rope_parameters"], rope = rope, {}
        full = rope.get("full_attention") or {"rope_type": "default"}
        local = rope.get("sliding_attention") or {"rope_type": "default"}
        full.update(c.get("rope_scaling") or {})
        full.setdefault("rope_theta", c.get("rope_theta", 1e6))
        local.setdefault("rope_theta", c.get("rope_local_base_freq", 1e4))
        tables = {}
        for t, p in (("full_attention", full), ("sliding_attention", local)):
            if t not in (layer_types or ()):
                continue
            kind = p.get("rope_type", p.get("type", "default"))
            theta, factor = p.get("rope_theta"), p.get("factor") if kind == "linear" else 1.0
            if set(p) - {"rope_type", "type", "rope_theta", "factor"} or kind not in ("default", "linear") \
                    or not _positive_real(theta) or not _positive_real(factor):
                bad[f"rope_parameters[{t}]"] = p
            else:
                tables[t] = (float(theta), float(factor))
        if bad:
            raise ValueError(f"unsupported Gemma 3 settings: {bad}")
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=n, num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"], rms_norm_eps=c["rms_norm_eps"],
            sliding_window=c["sliding_window"], layer_types=layer_types, rope=tables, tie_word_embeddings=True,
            max_position_embeddings=c["max_position_embeddings"],
        )
