import json
import random
import sys
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

# Sarvam-Translate's 22 Indian languages and English, a short phrase each (Manipuri in both of its scripts)
PHRASES = [
    "Be the change you wish to see in the world.",  # English
    "भारत एक विशाल और विविधतापूर्ण देश है।",  # Hindi
    "महाराष्ट्राची राजधानी मुंबई आहे.",  # Marathi
    "नेपाल हिमालयको काखमा रहेको सुन्दर देश हो।",  # Nepali
    "वसुधैव कुटुम्बकम्। सत्यमेव जयते॥",  # Sanskrit
    "अहाँ केहन छी? हम नीक छी।",  # Maithili
    "तुस केह् हाल ओ? मैं ठीक आं।",  # Dogri
    "आं बर' रावखौ बुङो।",  # Bodo
    "म्हजें नांव राम. तुमी कशें आसात?",  # Konkani
    "আমি বাংলায় কথা বলতে ভালোবাসি।",  # Bengali
    "অসমৰ ৰাজধানী দিছপুৰ। মই অসমীয়া কওঁ।",  # Assamese
    "ਪੰਜਾਬੀ ਮੇਰੀ ਮਾਂ ਬੋਲੀ ਹੈ।",  # Punjabi
    "ગુજરાતી ભાષા ખૂબ સુંદર છે.",  # Gujarati
    "ଓଡ଼ିଆ ଏକ ପ୍ରାଚୀନ ଭାଷା।",  # Odia
    "தமிழ் உலகின் பழமையான மொழிகளில் ஒன்று.",  # Tamil
    "తెలుగు భాష చాలా తీయనిది.",  # Telugu
    "ಕನ್ನಡ ನಮ್ಮ ಹೆಮ್ಮೆಯ ಭಾಷೆ.",  # Kannada
    "മലയാളം എന്റെ മാതൃഭാഷയാണ്.",  # Malayalam
    "ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ ᱚᱞ ᱪᱤᱠᱤ ᱛᱮ ᱚᱞᱚᱜ ᱠᱟᱱᱟ᱾",  # Santali (Ol Chiki)
    "ꯃꯤꯇꯩꯂꯣꯟ ꯑꯁꯤ ꯑꯩꯒꯤ ꯃꯃꯥ ꯂꯣꯟꯅꯤ꯫",  # Manipuri (Meitei Mayek)
    "মণিপুরী লোন",  # Manipuri (Bengali script)
    "اردو ایک خوبصورت زبان ہے۔",  # Urdu
    "کٲشُر زَبان چھےٚ واریاہ پرٲنۍ۔",  # Kashmiri
    "سنڌي ٻولي تمام پراڻي ۽ مٺي آهي.",  # Sindhi
]
# Unicode blocks of those scripts (combining marks included), and of scripts the vocabulary mostly lacks
BLOCKS = [(0x0900, 0x0980), (0xA8E0, 0xA900), (0x0980, 0x0A00), (0x0A00, 0x0A80), (0x0A80, 0x0B00), (0x0B00, 0x0B80),
          (0x0B80, 0x0C00), (0x0C00, 0x0C80), (0x0C80, 0x0D00), (0x0D00, 0x0D80), (0x1C50, 0x1C80), (0xABC0, 0xAC00),
          (0xAAE0, 0xAB00), (0x0600, 0x0700), (0x0750, 0x0780), (0x0020, 0x007F)]
SEPARATORS = [" ", "  ", "\u200c", "\u200d"]


def sarvam_cases(model_dir: str) -> list[str]:
    """For Gemma 3's SentencePiece-style BPE: every script above with byte fallback, whitespace (spaces become "▁"
    after the added-token split; runs of tabs, newlines and typed "▁" are added tokens), chat specials, typed
    byte-token text, characters outside the vocabulary, a 20,000-character chunk and rendered translation prompts."""
    from transformers import AutoTokenizer

    rng = random.Random(2)
    cases = list(PHRASES)
    for lo, hi in BLOCKS:
        chars = [chr(c) for c in range(lo, hi) if unicodedata.category(chr(c)) != "Cn"]
        for _ in range(8):
            cases.append("".join(rng.choice(chars) if rng.random() < 0.8 else rng.choice(SEPARATORS)
                                 for _ in range(rng.randint(1, 48))))
    cases += [
        "a  b   c    d", " ", "  ", " " * 31, " " * 32, " " * 70, "\t", "\t" * 31, "\t" * 40, "\n" * 31, "\n" * 45,
        "x \t \n y", " \n ", "\n \n", "\t \t", "a\u00a0b\u3000c\u2009d", "  hello world  ", " नमस्ते ", "\tindent\n",
        "\n\nstart", "line1\r\nline2\r\n", "\r", "\r\r\n\n", "हिन्दी\r\nதமிழ்\r\n",
        "▁hello", "▁▁▁", "a▁b", "▁ ▁", "▁" * 33,
        "👍🏽 👩🏾\u200d💻 👨\u200d👩\u200d👧\u200d👦 🏳️\u200d🌈 ❤️\u200d🔥 🇮🇳 🙏🏻", "🧑🏽\u200d🤝\u200d🧑🏿 #️⃣ 1️⃣",
        "क्\u200dष क्\u200cष र्\u200dय", "ന്\u200d ണ്\u200d ര്\u200d", "ক্\u200cষ", "می\u200cخواهم",
        "\u200d", "\u200c\u200c",
        "<start_of_turn>user\nTranslate the text below to Tamil.\n\nGood morning<end_of_turn>\n<start_of_turn>model\n",
        "<bos>", "<eos>", "<bos><bos><eos>", "<start_of_turn", "<end_of_turn>>", "<<start_of_turn>>",
        "<pad><unk><mask>[multimodal]", "<unused0><unused99><unused6241><unused6242>",
        "<image_soft_token><start_of_image><end_of_image>", "<table><tr><td>नमस्ते</td></tr></table>",
        "<h1>Title</h1>\n<b>bold</b>", "<0xEA>", "<0x0A>", "<0xea>", "<0xE0><0xA4><0xB9>", "<0x+F>",
        "<0xEA><start_of_turn><0xAF>",
        "𓀀𓀁 𐀀 𗀀 𠀀 ᚠᚢ 𐌰𐌱", "\x00\x07\x0b\x0c\x1b", "\ue000\uf8ff", "⠁⠃⠉", "ꫠꫡ ꣠꣡", "\U000E0041", "\U0010FFFF",
        "\u0378", "Hindi: नमस्ते, Tamil: வணக்கம், Urdu: سلام, Meitei: ꯈꯨꯔꯨꯝꯖꯔꯤ",
        "१२३ ১২৩ ੧੨੩ ૧૨૩ ୧୨୩ ௧௨௩ ౧౨౩ ೧೨೩ ൧൨൩ ᱑᱒᱓ ꯱꯲꯳ ۱۲۳",
    ]
    unspaced = "".join(PHRASES).replace(" ", "")
    cases.append((unspaced * (20000 // len(unspaced) + 1))[:20000])
    chat = AutoTokenizer.from_pretrained(model_dir)
    for messages in ([{"role": "system", "content": "Translate the text below to Hindi."},
                      {"role": "user", "content": PHRASES[0]}],
                     [{"role": "system", "content": "Translate the text below to English."},
                      {"role": "user", "content": "  " + PHRASES[19] + "\n"},
                      {"role": "assistant", "content": "Manipuri is my mother tongue."},
                      {"role": "user", "content": PHRASES[14]}]):
        cases.append(chat.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return cases


model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3.5-0.8b"
out_path = sys.argv[2] if len(sys.argv) > 2 else "tests/golden_tokens_qwen35.json"
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
cases = [
    "The capital of France is",
    "Hello world",
    "hello   world  ",
    "I'm here, you're there. Don't stop!",
    "Numbers: 12345 and 3.14159, year 2026",
    "Kaustubh builds an inference engine in TypeScript.",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "नमस्ते दुनिया — こんにちは世界 — 🚀🔥",
    "def f(x):\n    return x * 2\n\n\n",
    "   leading spaces and\ttabs\tand trailing   ",
    "",
    # scripts with combining marks (Qwen3.5's pre-tokenizer lists \p{M}; see src/tokenizer.py)
    "தமிழ் மொழி", "สวัสดีครับ", "مَرْحَبًا بِكُمْ", "שָׁלוֹם", "Tiếng Việt có dấu", "éclair", "❤️ 👍🏽",
    "ক্ষমা করুন", "ಕನ್ನಡ ಭಾಷೆ", "ພາສາລາວ", "Ελληνικά ά", "ǟb", "Zalgo z̷̢a̶l̵g̸o", "日本語のテキスト", "한국어 텍스트",
]
random.seed(0)
alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789.,!?'\n-_()[]{}<>|éü你好😀्ािु"
for _ in range(40):
    cases.append("".join(random.choice(alphabet) for _ in range(random.randint(1, 60))))
# digit grouping (Tiny Aya splits digits in threes from the right), other numeral systems, NFD, CRLF, chat specials
cases += [
    "12345", "1234567 apples", "12345abc", "x_12345", "a1b22c333d4444", "year 2026, pop 8,100,000", "price: $1,000,000.00",
    "١٢٣٤٥ عربي", "१२३४५ हिन्दी", "１２３４５", "日本語 123456 テキスト", "🚀🔥 2026", "éclair (NFD)",
    "line1\r\nline2\n\n\tend", "Kiswahili ni lugha", "Ọmọ Yorùbá", "ሰላም ልዑል", "Xin chào thế giới",
    "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi<|END_OF_TURN_TOKEN|>",
    # word boundaries after digits where the regex engines' \w differ (ZWNJ / ZWJ; superscripts, fractions)
    "10000\u200c", "۱۲۰۰۰\u200cتومان", "دهه ۱۹۷۰\u200cها", "area 1000² m", "15000½ kg", "2024\u200d",
]
random.seed(1)
digits = "0123456789٠١٢٣٤٥٦٧٨٩०१२३४५६७८९０１２ ,.ab_\n"
for _ in range(100):
    cases.append("".join(random.choice(digits) for _ in range(random.randint(1, 40))))
# where the merge table is not monotone (a merge ranked before the merge that makes one of its parts), the order in
# which pairs merge decides the answer: Qwen3.5 has 27 such merges, and these two words hit one (M6)
cases += ["いっぱいっぱい", "もういっぱいっぱいです"]
if Path(model_dir).name == "sarvam-translate":
    cases += sarvam_cases(model_dir)

out = []
for c in cases:
    plain = tok.encode(c, add_special_tokens=False).ids
    out.append({"text": c, "ids": tok.encode(c).ids, "ids_plain": plain, "decoded": tok.decode(plain, skip_special_tokens=False)})
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(out)} cases to {out_path}")
