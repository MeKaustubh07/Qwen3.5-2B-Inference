import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.utils import logging as hf_logging

from config import Gemma3Config
from models.gemma3 import Gemma3Model
from tiny_cohere2 import hf_greedy, hf_run, rel  # noqa: F401  (re-exported for the Gemma 3 tests)
from weight_loader import SafetensorsFile

hf_logging.disable_progress_bar()

LOCAL_THETA, GLOBAL_THETA, FACTOR = 10_000.0, 1_000_000.0, 8.0          # Gemma 3's own
FULL_LAYERS = [5, 11]
QK_NORM = 0.5          # q/k norm scales 1.5 +- 0.3: sharp attention, rounding (ops vs HF formulas) still < 1e-5


def hf_config(window: int, max_pos: int = 8192, eps: float = 1e-6) -> Gemma3TextConfig:
    """12 layers, so the 5:1 pattern repeats; 8 query heads per 2 KV heads of 32 dims (head_dim != hidden / heads,
    as in the real model); local theta != global, linear factor 8 on the full layers."""
    return Gemma3TextConfig(
        vocab_size=512, hidden_size=128, intermediate_size=320, num_hidden_layers=12, num_attention_heads=8,
        num_key_value_heads=2, head_dim=32, query_pre_attn_scalar=32, sliding_window=window,
        max_position_embeddings=max_pos, rms_norm_eps=eps, initializer_range=0.3, pad_token_id=0, bos_token_id=2,
        eos_token_id=1, rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": LOCAL_THETA},
                                         "full_attention": {"rope_type": "linear", "factor": FACTOR,
                                                            "rope_theta": GLOBAL_THETA}})


def build(tmp: str, window: int, seed: int = 0, max_pos: int = 8192, eps: float = 1e-6):
    """A random tiny Gemma 3, saved by HF's save_pretrained (gemma3_text config, tensors under "model.") and loaded by
    both. Large weights, so attention is far from uniform and a norm or RoPE mistake cannot hide in the noise."""
    torch.manual_seed(seed)
    hf = Gemma3ForCausalLM._from_config(hf_config(window, max_pos, eps), attn_implementation="eager").eval()
    with torch.no_grad():
        for name, m in hf.named_modules():            # RMSNorm weights start at 0 (a scale of 1): make them matter,
            if type(m).__name__ == "Gemma3RMSNorm":   # and q/k norms large, for sharp attention
                m.weight.copy_(QK_NORM * name.endswith(("q_norm", "k_norm")) + 0.3 * torch.randn_like(m.weight))
        hf.model.embed_tokens.weight.normal_(0, 0.05)   # a small tied head: greedy continuations vary (at 0.3, one
    hf.save_pretrained(tmp)                              # token repeats)
    return hf, lambda **kw: Gemma3Model(Gemma3Config.from_json(f"{tmp}/config.json"),
                                        SafetensorsFile(f"{tmp}/model.safetensors"), **kw)
