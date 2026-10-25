import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

MASKS = {"full_attention": create_causal_mask, "sliding_attention": create_sliding_window_causal_mask}


def fp32(t: torch.Tensor) -> torch.Tensor:
    """A fresh fp32 copy. Never a view of the file: a single-row product (a decode step) rounds differently on memory
    that is not 64-byte aligned, as an fp32 file's tensors may be."""
    return t.to(torch.float32, copy=True)


class StreamedDecoder:
    layer_class: type                     # the family's decoder layer, built as layer_class(config, i)
    layer_kwargs: dict = {}               # what the family's HF model passes each layer besides the tensors

    def __init__(self, model_dir: str, prefixes: tuple[str, ...] = ("model.",)):
        """prefixes: where the text model's tensors may sit in the checkpoint, in order of preference."""
        d = Path(model_dir)
        self.config = AutoConfig.from_pretrained(d).get_text_config()
        self.config._attn_implementation = "eager"
        index = d / "model.safetensors.index.json"
        weight_map = json.load(open(index))["weight_map"] if index.exists() else None
        files = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        self.files = {f: safe_open(str(d / f), framework="pt") for f in files}
        self.where = weight_map or {n: files[0] for n in self.files[files[0]].keys()}
        self.prefix = next((p for p in prefixes if p + "embed_tokens.weight" in self.where), None)
        if self.prefix is None:
            raise ValueError(f"no text model in {d} (no embed_tokens.weight under {' or '.join(prefixes)})")
        if not self.config.tie_word_embeddings or any(n.endswith("lm_head.weight") for n in self.where):
            raise ValueError("the answer key reads the head from embed_tokens; this checkpoint has a separate lm_head")

    def _get(self, name: str) -> torch.Tensor:
        """A text-model tensor by its name under the prefix ("layers.0.mlp.up_proj.weight")."""
        full = self.prefix + name
        return self.files[self.where[full]].get_tensor(full)

    def _slice(self, name: str):
        full = self.prefix + name
        return self.files[self.where[full]].get_slice(full)

    def _loaded(self, module: torch.nn.Module, name: str) -> torch.nn.Module:
        """module with its weight set to the tensor `name`, widened to fp32."""
        module.weight.data = fp32(self._get(name))
        return module

    def layer(self, i: int) -> torch.nn.Module:
        """Decoder layer i with its weights widened to fp32 (built on the meta device, so nothing is initialised)."""
        prefix = f"{self.prefix}layers.{i}."
        with torch.device("meta"):
            layer = self.layer_class(self.config, i)
        weights = {n[len(prefix):]: fp32(self.files[self.where[n]].get_tensor(n))
                   for n in self.where if n.startswith(prefix)}
        layer.load_state_dict(weights, strict=True, assign=True)
        return layer.eval()

    def embed(self, ids: list[int]) -> torch.Tensor:
        table = self._slice("embed_tokens.weight")
        return torch.stack([table[i:i + 1][0] for i in ids]).float()

    def tied_head(self, h: torch.Tensor, rows: int) -> torch.Tensor:
        """[N, hidden] -> [N, vocab] through the embedding table, widened `rows` rows at a time."""
        V = self.config.vocab_size
        table = self._slice("embed_tokens.weight")
        return torch.cat([h @ fp32(table[r:min(r + rows, V)]).T for r in range(0, V, rows)], dim=-1)

    # ---------------------------------------------------------------- family hooks
    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Final-norm hidden states [N, hidden] -> logits [N, vocab]."""
        raise NotImplementedError

    def position_embeddings(self, h: torch.Tensor, position_ids: torch.Tensor) -> dict:
        """layer type -> the (cos, sin) its layers get, as the family's HF model computes them."""
        raise NotImplementedError

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def run_all(self, batch: list[list[int]], caches: list[DynamicCache],
                captures: list[dict | None] | None = None) -> list[torch.Tensor]:
        """Each sequence's new tokens, continuing its own cache -> its final-norm hidden states [T, hidden], as the HF
        text model's forward computes them, with each decoder layer built once for all the sequences."""
        c = self.config
        types = set(c.layer_types)
        captures = captures or [None] * len(batch)
        hs, ctx = [], []
        for ids, cache, cap in zip(batch, caches, captures):       # masks from the caches BEFORE any layer updates them
            start = cache.get_seq_length()
            h = self.embed(ids)[None]                                            # [1, T, hidden]
            position_ids = torch.arange(start, start + len(ids))[None]
            kw = dict(config=c, inputs_embeds=h, attention_mask=None, past_key_values=cache, position_ids=position_ids)
            hs.append(h)
            ctx.append(({t: MASKS[t](**kw) for t in types}, self.position_embeddings(h, position_ids), position_ids))
            if cap is not None:
                cap["embed"] = h[0].clone()
        for i in range(c.num_hidden_layers):
            layer, t = self.layer(i), c.layer_types[i]
            for j, (cache, (masks, pe, position_ids), cap) in enumerate(zip(caches, ctx, captures)):
                hs[j] = layer(hs[j], attention_mask=masks[t], position_embeddings=pe[t], past_key_values=cache,
                              position_ids=position_ids, **self.layer_kwargs)
                if cap is not None:
                    cap[f"l{i}_out"] = hs[j][0].clone()
            del layer
        out = [self.norm(h)[0] for h in hs]
        for h, cap in zip(out, captures):
            if cap is not None:
                cap["final_norm"] = h.clone()
        return out

    def greedy_all(self, prompts: list[list[int]], n: int, captures: list[dict | None] | None = None,
                   keep_from: list[int] | None = None, stop=(), log=None, on_done=None):
        """Prefill every prompt, then up to n argmax tokens each, all in lockstep; a prompt whose last token is in
        stop leaves the lockstep -> (per prompt the logits of its positions from keep_from[i] on (default: all), per
        prompt its new tokens (a stop token included), per prompt the logits of its decode steps
        [len(tokens) - 1, vocab]). on_done(i, logits, tokens, step_logits) is called as soon as prompt i is
        finished, so a caller can save it before the others are."""
        caches = [DynamicCache(config=self.config) for _ in prompts]
        hs = self.run_all(prompts, caches, captures)
        if log:
            log(f"prefill: {sum(map(len, prompts))} tokens of {len(prompts)} prompts")
        rows = [h[(keep_from or [0] * len(hs))[i]:] for i, h in enumerate(hs)]
        logits = list(self.head(torch.cat(rows)).split([len(r) for r in rows]))
        out, steps, live = [[int(lg[-1].argmax())] for lg in logits], [[] for _ in prompts], list(range(len(prompts)))

        def stacked(st):
            return torch.stack(st) if st else torch.empty(0)

        def settle():
            for j in [j for j in live if out[j][-1] in stop or len(out[j]) == n]:
                live.remove(j)
                if on_done:
                    on_done(j, logits[j], out[j], stacked(steps[j]))
        settle()
        while live:
            hs = self.run_all([[out[j][-1]] for j in live], [caches[j] for j in live])
            for j, lg in zip(live, self.head(torch.cat([h[-1:] for h in hs]))):
                out[j].append(int(lg.argmax()))
                steps[j].append(lg)
            if log:
                log(f"token {max(map(len, out))}/{n}" + (f", {len(live)} of {len(out)} prompts" if stop else ""))
            settle()
        return logits, out, [stacked(st) for st in steps]

    def greedy(self, ids: list[int], n: int, capture: dict | None = None) -> tuple[torch.Tensor, list[int]]:
        """One prompt: (logits of every position, n argmax tokens)."""
        logits, out, _ = self.greedy_all([ids], n, [capture])
        return logits[0], out[0]
