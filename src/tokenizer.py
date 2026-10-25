import heapq
import json
import unicodedata
from pathlib import Path

import regex  # third-party engine: needed for \p{L} / \p{N} classes that the stdlib `re` lacks

# HF tokenizers' ByteFallback decoder reads a token as a byte when it is 6 bytes long, "<0x" + 2 + ">", and the middle
# parses with Rust's u8::from_str_radix(.., 16), which also takes a leading "+".
_BYTE_TOKEN = regex.compile(r"<0x(?:[0-9A-Fa-f]{2}|\+[0-9A-Fa-f])>")

# HF tokenizers runs these regexes with Oniguruma, whose \w differs from the `regex` module's on 8 characters: ZWNJ
# and ZWJ are word characters only in `regex`; ² ³ ¹ ¼ ½ ¾ only in Oniguruma. Tiny Aya's digit split ends in \b, so
# a number written before a ZWNJ (common in Persian: ۱۹۷۰‌ها) would be grouped differently. \b is rewritten to
# Oniguruma's word boundary; \w, \W and \B would need the same and are refused.
_ONIG_W = r"(?:(?![\u200c\u200d])\w|[\u00b2\u00b3\u00b9\u00bc-\u00be])"
_ONIG_B = rf"(?:(?<={_ONIG_W})(?!{_ONIG_W})|(?<!{_ONIG_W})(?={_ONIG_W}))"


def oniguruma_compatible(pattern: str) -> str:
    r"""The pattern with every \b outside a character class replaced by Oniguruma's word boundary."""
    out, i, in_class = [], 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            e = pattern[i + 1]
            if e in "wWB" or (e == "b" and in_class):
                raise NotImplementedError(f"\\{e} in a pre-tokenizer regex: {pattern!r}")
            out.append(_ONIG_B if e == "b" else c + e)
            i += 2
            continue
        in_class = (in_class and c != "]") or (not in_class and c == "[")
        out.append(c)
        i += 1
    return "".join(out)


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
    the number of words (Gemma 3 has 6,415 added tokens, 6,322 of them starting with "<")."""
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


def _fallback_text(run: bytearray) -> str:
    """A run of byte tokens as HF's ByteFallback decodes it: its UTF-8 text, or one U+FFFD per byte if invalid."""
    try:
        return run.decode("utf-8")
    except UnicodeDecodeError:
        return "\ufffd" * len(run)


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
        self.byte_level = bool(steps) and steps[-1]["type"] == "ByteLevel"
        if self.byte_level:
            self._init_byte_level(spec, steps)
        else:
            self._init_sentencepiece(spec, steps)

        # After BPE: Tiny Aya's and Gemma 3's post-processor (TemplateProcessing) puts BOS in front of every sequence;
        # Qwen3.5's (ByteLevel) only adjusts offsets. A Sequence may hold ByteLevel steps and one template.
        pp = spec.get("post_processor")
        procs = (pp["processors"] if pp["type"] == "Sequence" else [pp]) if pp else []
        templates = [x for x in procs if x["type"] == "TemplateProcessing"]
        if len(templates) > 1 or any(x["type"] not in ("ByteLevel", "TemplateProcessing") for x in procs):
            raise NotImplementedError(f"post-processor {pp}")
        single = templates[0]["single"] if templates else [{"Sequence": {"id": "A"}}]
        if [next(iter(x)) for x in single] == ["SpecialToken", "Sequence"]:
            self.bos_id: int | None = templates[0]["special_tokens"][single[0]["SpecialToken"]["id"]]["ids"][0]
        elif [next(iter(x)) for x in single] == ["Sequence"]:
            self.bos_id = None
        else:
            raise NotImplementedError(f"post-processor template {single}")

    def _init_byte_level(self, spec: dict, steps: list[dict]) -> None:
        # Stage 1, normalization: Qwen3.5's file asks for NFC; Tiny Aya's has none (text is used as given).
        norm = spec["normalizer"]
        if norm not in (None, {"type": "NFC"}):
            raise NotImplementedError(f"normalizer {norm}")
        self.nfc = norm is not None

        # Stage 2, pre-tokenization: a Sequence of regex Splits, then ByteLevel. Each Split is "Isolated": a match
        # becomes its own piece and the text between matches stays as pieces too. Qwen3.5 has one Split whose regex
        # covers every character; Tiny Aya first splits off digit groups (3 at a time, from the right), then splits
        # every piece with a GPT-4o-style regex. ByteLevel with use_regex=false only maps bytes to stand-in chars.
        # The regexes are taken from the file verbatim (the `regex` engine supports (?i:...)), except that \b gets
        # Oniguruma's meaning (see oniguruma_compatible). Known, rare differences from HF that remain: `regex` uses
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
        self.splits = [regex.compile(oniguruma_compatible(p["pattern"]["Regex"])) for p in splits]

        if spec["model"].get("byte_fallback"):
            raise NotImplementedError("byte_fallback in a byte-level BPE")
        self.byte_to_char, self.char_to_byte = build_byte_alphabet()

    def _init_sentencepiece(self, spec: dict, steps: list[dict]) -> None:
        # Gemma 3: the normalizer writes each space as "▁", so the Split on " " that follows finds nothing and a whole
        # text between added tokens is one BPE chunk. Accepted only where that holds: a Split on the single character
        # the normalizer replaces, which its replacement does not contain.
        norm, model = spec["normalizer"], spec["model"]
        if not norm or norm["type"] != "Replace" or "String" not in norm["pattern"]:
            raise NotImplementedError(f"normalizer {norm}")
        self.replace = (norm["pattern"]["String"], norm["content"])
        for p in steps:
            if p["type"] != "Split" or p["invert"] or p["pattern"] != {"String": self.replace[0]} \
                    or len(self.replace[0]) != 1 or self.replace[0] in self.replace[1]:
                raise NotImplementedError(f"pre-tokenizer step {p} after normalizer {norm}")
        self.nfc = False

        # merge_word in HF tokenizers' BPE: byte fallback writes a character as its <0xXX> tokens only when the
        # vocabulary has every one of them; otherwise the character is the unknown token, or dropped without one.
        self.unk, self.fuse_unk = model.get("unk_token"), bool(model.get("fuse_unk"))
        self.byte_symbols = [t if (t := f"<0x{b:02X}>") in self.vocab else None for b in range(256)] \
            if model.get("byte_fallback") else None

        dec = spec["decoder"]
        parts = dec["decoders"] if dec and dec["type"] == "Sequence" else []
        if [d["type"] for d in parts] != ["Replace", "ByteFallback", "Fuse"] or "String" not in parts[0]["pattern"]:
            raise NotImplementedError(f"decoder {dec}")
        self.unreplace = (parts[0]["pattern"]["String"], parts[0]["content"])
        self.fallback_byte: dict[int, int] = {}       # HF decodes added tokens too, so any token can read as a byte
        for i, t in self.id_to_token.items():
            if _BYTE_TOKEN.fullmatch(t := t.replace(*self.unreplace)):
                self.fallback_byte[i] = int(t[3:5], 16)

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

    # ---------- Stage 2, SentencePiece-style: the symbols BPE starts from ----------
    def _word(self, text: str) -> list[str]:
        """merge_word in HF tokenizers' BPE model: a character in the vocabulary is a symbol; any other is its byte
        tokens, else the unknown token (one per run with fuse_unk). Like HF, a pending unknown token is written only
        when an in-vocabulary character or the end comes, so it lands after byte tokens that come between."""
        vocab, fallback, unk, out, pending = self.vocab, self.byte_symbols, self.unk, [], False
        for c in text:
            if c in vocab:
                if pending:
                    out.append(unk)
                    pending = False
                out.append(c)
            elif fallback and None not in (b := [fallback[x] for x in c.encode("utf-8")]):
                out += b
            elif unk is not None:
                if pending and not self.fuse_unk:
                    out.append(unk)
                pending = True
        if pending:
            out.append(unk)
        return out

    # ---------- Stage 3: BPE merging for ONE chunk ----------
    def _bpe(self, chunk: str | list[str]) -> list[str]:
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
    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """add_bos: put the model's BOS in front (only if its post-processor has one), as for a raw prompt.
        Off by default: a chat template writes BOS itself, and a continuation (a decision option, a next turn)
        must never get one."""
        ids: list[int] = [self.bos_id] if add_bos and self.bos_id is not None else []
        for part in self.special_re.split(text) if self.special_re else [text]:   # Stage 0: peel off specials
            if not part:
                continue
            if part in self.special:
                ids.append(self.special[part])
                continue
            if not self.byte_level:
                ids += [self.vocab[p] for p in self._bpe(self._word(part.replace(*self.replace)))]
                continue
            if self.nfc:
                part = unicodedata.normalize("NFC", part)                # Stage 1
            for chunk in self._pieces(part):                              # Stage 2: word-like chunks
                mapped = "".join(self.byte_to_char[b] for b in chunk.encode("utf-8"))  # Stage 2b
                for piece in self._bpe(mapped):                           # Stage 3
                    ids.append(self.vocab[piece])                         # vocab lookup
        return ids

    # ---------- Decode: ids -> text ----------
    def decode(self, ids: list[int]) -> str:
        if not self.byte_level:
            return self._decode_sentencepiece(ids)
        out = bytearray()
        for i in ids:
            tok = self.id_to_token[i]
            if tok in self.special:
                out += tok.encode("utf-8")
            else:
                out += bytes(self.char_to_byte[c] for c in tok)           # stand-in char -> byte
        return out.decode("utf-8", errors="replace")

    def _decode_sentencepiece(self, ids: list[int]) -> str:
        """The decoder Sequence[Replace, ByteFallback, Fuse] over every token: "▁" back to " ", each run of byte
        tokens joined into UTF-8, everything concatenated."""
        out, run = [], bytearray()
        for i in ids:
            if (b := self.fallback_byte.get(i)) is not None:
                run.append(b)
                continue
            if run:
                out.append(_fallback_text(run))
                run.clear()
            out.append(self.id_to_token[i].replace(*self.unreplace))
        if run:
            out.append(_fallback_text(run))
        return "".join(out)

    def is_final(self, text: str, last_id: int) -> bool:
        """Whether `text`, decoded from ids ending in last_id, stays the start of the decode however the ids go on,
        so a stream may send it. Byte-level: unless it ends in U+FFFD, a character cut between tokens. SentencePiece:
        unless last_id is a byte token, since ByteFallback decodes a run all or nothing: a character already decoded
        turns into U+FFFD if the run goes on with bytes that make it invalid."""
        if self.byte_level:
            return not text.endswith("\ufffd")
        return last_id not in self.fallback_byte

    def vocab_size(self) -> int:
        return len(self.id_to_token)
