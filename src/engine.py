import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from backend.metal import MetalBackend
from backend.torch_ref import TorchBackend
from chat import TemplateChat
from config import Cohere2Config, Gemma3Config, Qwen35Config
from models.cohere2 import Cohere2Model
from models.gemma3 import Gemma3Model
from models.qwen3_5 import Qwen35Model
from quant import QtFile
from tokenizer import Tokenizer
from weight_loader import open_weights

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("engine")

MODELS = {
    "qwen3.5-0.8b": dict(dir="models/qwen3.5-0.8b", family="qwen3_5", weights="model.safetensors-00001-of-00001.safetensors",
                         eos=[248046, 248044], chat="qwen3.5", thinking=True, policy="configs/quant/qwen3.5-0.8b.json",
                         sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)),
    "qwen3.5-2b": dict(dir="models/qwen3.5-2b", family="qwen3_5", weights="model.safetensors-00001-of-00001.safetensors",
                       eos=[248046, 248044], chat="qwen3.5", thinking=True, policy="configs/quant/qwen3.5-2b.json",
                       sampling=dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)),
    # Tiny Aya Global (Cohere2). chat "template": the model's own Jinja template, read from its tokenizer_config.json.
    # Stops: <EOS_TOKEN>, <|END_OF_TURN_TOKEN|>, <|END_RESPONSE|>. Sampling: the model card's. max_model_len: its
    # config's 8192 (prompt + output), checked past the 4096-token window (tests/test_aya_long*.py). No
    # thinking mode. policy: written by scripts/calibrate_quant.py; until then INT4 warns. backends: only the
    # quantized Metal ones fit in 8 GB (fp32 on the CPU would need 13.4 GB, bf16 on the GPU 6.7 GB). model_kwargs: what
    # the model class is built with; kv_dtype: its KV is 6x Qwen3.5-2B's per token (36 layers x 4 heads x 128), so it
    # is stored in bf16 (Qwen3.5 keeps fp32).
    # kv_blocks: the server's pool, 768 blocks of every layer = 864 MiB of bf16 KV (two 8K sequences, rings counted);
    # with the INT4 weights 8K prompts then hold 4.45 GiB of GPU memory. per_backend: what one backend changes. INT8's
    # weights are 1.2 GB larger, and at 8K it held 5.48 GiB, over the 5.33 GiB Metal recommends on an 8 GB Mac, where
    # another GPU user can make Metal abort work silently. So INT8 serves at most 4096 tokens from 512 blocks (two 4K
    # sequences): 4.45 GiB generating, 4.82 scoring /v1/decide options, whose context lives outside the pool
    # (with 768 blocks, 5.21). INT4 /v1/decide holds its context outside the pool, and the prefill of an 8K one peaks
    # at 4.99 GiB, so INT4 decides take at most 6144 tokens (4.60 GiB); generation keeps 8192 (docs/results-log.md, M5
    # part C and M6).
    "tiny-aya-global": dict(dir="models/tiny-aya-global", family="cohere2", weights="model.safetensors.index.json",
                            eos=[3, 6, 261001], chat="template", policy="configs/quant/tiny-aya-global.json",
                            sampling=dict(temperature=0.1, top_p=0.95, top_k=50, repetition_penalty=1.0),
                            max_model_len=8192, backends=("metal-int8", "metal-int4"),
                            model_kwargs=dict(kv_dtype=torch.bfloat16),
                            kv_blocks=768, per_backend={"metal-int8": dict(max_model_len=4096, kv_blocks=512),
                                                         "metal-int4": dict(decide_max_len=6144)}),
    # Sarvam-Translate (Gemma 3 4B text model; docs/sarvam-translate-plan.md). Stops: <eos>, <end_of_turn>
    # (generation_config.json). Sampling: the card's temperature 0.01 with generation_config's top_k / top_p.
    # max_model_len: the card's 8K (the config's 131072 is Gemma 3's). backends: quantized Metal only (the text model
    # is 15.5 GB in fp32, 7.8 GB in bf16). model_kwargs: bf16 KV (136 KiB per token over 34 layers); the embedding
    # scaled by sqrt(2560) rounded to bf16 (50.5), as transformers computes it with the checkpoint's bf16 weights.
    # kv_blocks: the server's pool, 320 blocks of every layer = 680 MiB of bf16 KV (two 8K sequences: 317, the sliding
    # layers' rings counted). GPU peaks measured as the server runs (forward_packed in 512-token chunks, then
    # decode_batch; three runs in one process; Metal's recommended working set 5.33 GiB, kept under 0.95x): INT4 (the
    # calibrated policy, 2.99 GB) with an 8K prompt 4.40 GiB. INT8's weights are 4.12 GB: an 8K prompt beside a
    # 160-block pool reached 5.05 GiB on its second run, so INT8 serves at most 4096 tokens from 241 blocks (two 4K
    # sequences): 4.79 GiB. /v1/decide holds its context outside the pool (16 options x 64 tokens): INT4 at 8K 5.47
    # GiB, at 6K 4.84, so INT4 decides take at most 6144 tokens; INT8 at 4K 5.34, 3K 5.14, 2K 4.97, so at most 2048.
    "sarvam-translate": dict(dir="models/sarvam-translate", family="gemma3", weights="model.safetensors.index.json",
                             eos=[1, 106], chat="template", policy="configs/quant/sarvam-translate.json",
                             sampling=dict(temperature=0.01, top_p=0.95, top_k=64, repetition_penalty=1.0),
                             max_model_len=8192, backends=("metal-int8", "metal-int4"),
                             model_kwargs=dict(kv_dtype=torch.bfloat16, embed_scale_dtype=torch.bfloat16),
                             kv_blocks=320, per_backend={"metal-int8": dict(max_model_len=4096, kv_blocks=241,
                                                                            decide_max_len=2048),
                                                         "metal-int4": dict(decide_max_len=6144)}),
}
FAMILIES = {"qwen3_5": (Qwen35Config, Qwen35Model),          # family -> (config class, model class)
            "cohere2": (Cohere2Config, Cohere2Model),
            "gemma3": (Gemma3Config, Gemma3Model)}


@dataclass
class Engine:
    name: str
    model: object
    tokenizer: Tokenizer
    eos_ids: set[int]
    chat_style: str | TemplateChat
    sampling: dict = field(default_factory=dict)
    max_model_len: int | None = None           # a model's hard cap on prompt + generated tokens (None: no cap)
    thinking: bool = False                     # the model has a reasoning mode (Qwen3.5: <think> ... </think>)
    kv_blocks: int | None = None               # the server's default KV pool, in blocks of every layer (None: 1024)
    decide_max_len: int | None = None          # /v1/decide's limit on context + option, if below max_model_len


def chat_style(spec: dict, d: Path) -> str | TemplateChat:
    return TemplateChat(str(d / "tokenizer_config.json")) if spec["chat"] == "template" else spec["chat"]


def make_backend(kind: str, policy: str | None):
    if kind == "cpu":
        return TorchBackend("cpu", torch.float32)
    if kind == "mps":
        return TorchBackend("mps", torch.bfloat16)
    if kind == "metal":
        return MetalBackend()
    if kind in ("metal-int8", "metal-int4"):
        scheme = kind.split("-")[1]
        if scheme == "int4" and policy and not (ROOT / policy).exists():
            log.warning(f"no INT4 policy {policy}: every group goes to INT4 (scripts/calibrate_quant.py writes it)")
        return MetalBackend(scheme, policy if scheme == "int4" and policy and (ROOT / policy).exists() else None)
    raise ValueError(f"unknown backend {kind}")


def load_engine(name: str, backend: str = "metal", weights_file: str | None = None) -> Engine:
    spec = MODELS[name] | MODELS[name].get("per_backend", {}).get(backend, {})
    d = ROOT / spec["dir"]
    if spec["family"] not in FAMILIES:
        raise NotImplementedError(f"{name}: no model class for family {spec['family']!r} yet")
    if "backends" in spec and backend not in spec["backends"]:
        raise ValueError(f"{name} runs on {' or '.join(spec['backends'])} on this machine, not {backend}")
    config_cls, model_cls = FAMILIES[spec["family"]]
    cfg, tok = config_cls.from_json(str(d / "config.json")), Tokenizer(str(d / "tokenizer.json"))
    # logits are cut to tokenizer.vocab_size() (the head may be padded past it): the ids must be contiguous, and every
    # stop id must survive the cut
    n = tok.vocab_size()
    if max(tok.id_to_token) + 1 != n or n > cfg.vocab_size or max(spec["eos"]) >= n:
        raise ValueError(f"{name}: {n} token ids (max {max(tok.id_to_token)}), {cfg.vocab_size} output rows, "
                         f"stop ids {spec['eos']}")
    be = make_backend(backend, str(ROOT / p) if (p := spec.get("policy")) else None)
    weights = QtFile(weights_file) if weights_file else open_weights(str(d / spec["weights"]))
    model = model_cls(cfg, weights, be, **spec.get("model_kwargs", {}))
    caps = [c for c in (spec.get("max_model_len"), getattr(model, "max_positions", None)) if c]
    return Engine(name, model, tok, set(spec["eos"]), chat_style(spec, d), spec["sampling"],
                  min(caps) if caps else None, spec.get("thinking", False), spec.get("kv_blocks"),
                  spec.get("decide_max_len"))
