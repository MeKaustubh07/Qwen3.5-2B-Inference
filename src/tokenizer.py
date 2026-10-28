import heapq
import json
import unicodedata
from pathlib import Path

import regex  # third-party engine: needed for \p{L} / \p{N} classes that the stdlib `re` lacks

def build_byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    """The 256 atoms: every byte value 0..255 gets a printable stand-in character.
    Printable bytes stand for themselves; the rest are shifted up to 256+ so they stay visible
    (space -> 'Ġ', newline -> 'Ċ'). This is GPT-2's bytes_to_unicode, which Qwen inherits."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    byte_to_char: dict[int, str] = {}
    nxt = 256
    for b in range(256):
        if b in printable:
            byte_to_char[b] = chr(b)
        else:
            byte_to_char[b] = chr(nxt)
            nxt += 1
    char_to_byte = {c: b for b, c in byte_to_char.items()}
    return byte_to_char, char_to_byte


def literal_alternation(words) -> str:
    """A regex matching any of `words` leftmost-longest, as HF's added-token split (Aho-Corasick) does: the words as
    a trie whose nodes try their continuations before ending there, so a text position costs the trie's depth, not
    the number of words."""
    trie: dict = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node[""] = {}

    def emit(node: dict) -> str:
        alts = [regex.escape(c) + emit(child) for c, child in node.items() if c]
        if "" in node:
            alts.append("")                            # a word ends here: tried after every longer one
        return alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"

    return emit(trie)


class Tokenizer:
    def __init__(self, path: str):
        spec = json.load(open(path, encoding="utf-8"))
        model = spec["model"]
        options = {k: v for k, v in model.items() if k not in ("vocab", "merges")}
        unsupported = ("ignore_merges", "dropout", "continuing_subword_prefix", "end_of_word_suffix")
        if model["type"] != "BPE" or any(options.get(k) for k in unsupported):
            raise NotImplementedError(f"BPE model {options}")

        self.vocab: dict[str, int] = model["vocab"]                       # token string -> id
        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}

        # merges are either "a b" strings or [a, b] pairs depending on file version; rank = position
        self.merge_rank: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(model["merges"]):
            a, b = m.split(" ") if isinstance(m, str) else m
            self.merge_rank[(a, b)] = rank

        if any(t.get(f) for t in spec["added_tokens"] for f in ("lstrip", "rstrip", "single_word", "normalized")):
            raise NotImplementedError("added tokens with lstrip / rstrip / single_word / normalized set")
        self.special: dict[str, int] = {t["content"]: t["id"] for t in spec["added_tokens"]}
        # Some checkpoints declare extra added tokens only in tokenizer_config.json (Qwen3.5: 248070..248076)
        cfg_path = Path(path).with_name("tokenizer_config.json")
        if cfg_path.exists():
            for tid, t in json.load(open(cfg_path, encoding="utf-8")).get("added_tokens_decoder", {}).items():
                self.special.setdefault(t["content"], int(tid))
        self.id_to_token.update({i: t for t, i in self.special.items()})
        self.special_re = regex.compile(f"({literal_alternation(self.special)})") if self.special else None

        pt = spec["pre_tokenizer"]
        steps = (pt["pretokenizers"] if pt["type"] == "Sequence" else [pt]) if pt else []
        if not steps or steps[-1]["type"] != "ByteLevel":
            raise NotImplementedError(f"pre-tokenizer {pt}: only byte-level BPE")
        self._init_byte_level(spec, steps)

        # After BPE: Qwen3.5's post-processor (ByteLevel) only adjusts offsets; one that adds tokens is refused
        pp = spec.get("post_processor")
        if pp and pp["type"] != "ByteLevel":
            raise NotImplementedError(f"post-processor {pp}")

    def _init_byte_level(self, spec: dict, steps: list[dict]) -> None:
        # Stage 1, normalization: Qwen3.5's file asks for NFC.
        norm = spec["normalizer"]
        if norm != {"type": "NFC"}:
            raise NotImplementedError(f"normalizer {norm}")

        # Stage 2, pre-tokenization: a Sequence of regex Splits, then ByteLevel. Each Split is "Isolated": a match
        # becomes its own piece and the text between matches stays as pieces too. Qwen3.5 has one Split whose regex
        # covers every character. ByteLevel with use_regex=false only maps bytes to stand-in chars. The regexes are
        # taken from the file verbatim (the `regex` engine supports (?i:...)). HF runs them with Oniguruma, whose \w
        # differs from the `regex` module's on 8 characters, so \b, \B, \w and \W are refused. Known, rare
        # differences from HF that remain: `regex` uses
        # Unicode 17 tables and Oniguruma Unicode 16, so ~4,700 code points new in Unicode 17 are letters or digits
        # only here; and Python's NFC (Qwen3.5) composes around a few dozen combining marks newer than tokenizers' NFC
        # tables, which treat them as starters.
        # Note: for Qwen3.5 checkpoints, transformers' AutoTokenizer resolves to Qwen2Tokenizer (per the checkpoint's
        # tokenizer_config.json) and substitutes the older Qwen2 regex, which lacks \p{M}; it then splits combining
        # marks (Hindi, Tamil, Thai, Arabic) differently from this file. The file and transformers' own
        # Qwen3_5Tokenizer agree with each other, and this engine follows them.
        *splits, last = steps
        if last["use_regex"] or last["add_prefix_space"]:
            raise NotImplementedError(f"pre-tokenizer must end with a plain ByteLevel step, got {last}")
        for p in splits:
            if p["type"] != "Split" or p["behavior"] != "Isolated" or p["invert"]:
                raise NotImplementedError(f"pre-tokenizer step {p}")
        for p in splits:
            if regex.search(r"\\[bBwW]", p["pattern"]["Regex"]):
                raise NotImplementedError(f"word-character class or boundary in a pre-tokenizer regex: {p}")
        self.splits = [regex.compile(p["pattern"]["Regex"]) for p in splits]

        if spec["model"].get("byte_fallback"):
            raise NotImplementedError("byte_fallback in a byte-level BPE")
        self.byte_to_char, self.char_to_byte = build_byte_alphabet()

    # ---------- Stage 2: regex pre-splits ----------
    def _pieces(self, text: str) -> list[str]:
        """Apply every Split in turn to every piece so far. Isolated: each match is its own piece, and the text
        between matches is kept as pieces too (a plain findall would silently drop it)."""
        pieces = [text]
        for rx in self.splits:
            out = []
            for s in pieces:
                last = 0
                for m in rx.finditer(s):
                    if m.start() > last:
                        out.append(s[last:m.start()])     # the gap before this match
                    if m.end() > m.start():
                        out.append(m.group())             # the match itself (empty matches add nothing)
                    last = m.end()
                if last < len(s):
                    out.append(s[last:])                  # the tail after the last match
            pieces = out
        return pieces

    # ---------- Stage 3: BPE merging for ONE chunk ----------
    def _bpe(self, chunk: str) -> list[str]:
        """Merge a chunk's characters as HF tokenizers does (merge_all in its BPE model): a min-heap of adjacent
        pairs keyed by (merge rank, position) over a doubly linked list of symbols, so a long chunk costs n log n,
        not n^2 (an unspaced CJK text is one chunk). Each pair is merged where it was found, one occurrence at a
        time: a lower-ranked pair formed by a merge goes before the remaining occurrences of the current one. A
        queued pair is skipped when its left symbol was absorbed, has no right neighbour, or no longer forms the
        token it was queued for; like HF, it is the token that is compared, not the rank."""
        sym: list[str | None] = list(chunk)
        n = len(sym)
        if n < 2:
            return sym
        nxt, prv, rank = list(range(1, n + 1)), list(range(-1, n - 1)), self.merge_rank
        nxt[-1] = -1
        heap = [(r, i, sym[i] + sym[i + 1]) for i in range(n - 1) if (r := rank.get((sym[i], sym[i + 1]))) is not None]
        heapq.heapify(heap)
        while heap:
            _, i, token = heapq.heappop(heap)
            j = nxt[i]
            if sym[i] is None or j == -1 or sym[i] + sym[j] != token or (sym[i], sym[j]) not in rank:
                continue                                       # stale: the pair it was queued for is gone
            sym[i], sym[j] = token, None                       # the right symbol is absorbed into the left
            k = nxt[i] = nxt[j]
            if k != -1:
                prv[k] = i
            if (p := prv[i]) != -1 and (r := rank.get((sym[p], token))) is not None:
                heapq.heappush(heap, (r, p, sym[p] + token))   # the new pair on the left
            if k != -1 and (r := rank.get((token, sym[k]))) is not None:
                heapq.heappush(heap, (r, i, token + sym[k]))   # and on the right
        return [s for s in sym if s is not None]

    # ---------- Encode: text -> ids ----------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self.special_re.split(text) if self.special_re else [text]:   # Stage 0: peel off specials
            if not part:
                continue
            if part in self.special:
                ids.append(self.special[part])
                continue
            part = unicodedata.normalize("NFC", part)                    # Stage 1
            for chunk in self._pieces(part):                              # Stage 2: word-like chunks
                mapped = "".join(self.byte_to_char[b] for b in chunk.encode("utf-8"))  # Stage 2b
                for piece in self._bpe(mapped):                           # Stage 3
                    ids.append(self.vocab[piece])                         # vocab lookup
        return ids

    # ---------- Decode: ids -> text ----------
    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            tok = self.id_to_token[i]
            if tok in self.special:
                out += tok.encode("utf-8")
            else:
                out += bytes(self.char_to_byte[c] for c in tok)           # stand-in char -> byte
        return out.decode("utf-8", errors="replace")

    def is_final(self, text: str, last_id: int) -> bool:
        """Whether `text`, decoded from ids ending in last_id, stays the start of the decode however the ids go on,
        so a stream may send it: unless it ends in U+FFFD, a character cut between tokens."""
        return not text.endswith("\ufffd")

    def vocab_size(self) -> int:
        return len(self.id_to_token)
