# Sarvam-Translate in 22 languages: this engine's INT4 vs Tiny Aya and Qwen3.5-2B

How well Sarvam-Translate (Gemma 3 4B, fine-tuned for translation) translates into and out of each of the 22 scheduled
Indian languages when served by this engine in INT4 on the MacBook Air M2 (8 GB), against the two general models the
engine already serves, Tiny Aya Global and Qwen3.5-2B, both in INT4 too. Generated 2026-10-24 22:17 to 2026-10-25
23:49 and scored 2026-10-25, one model in memory at a time. Raw output: [`raw/sarvam_translate/`](raw/sarvam_translate/)
(`tables.md`, `numbers.json`, `numbers.csv`, [`chrf.svg`](raw/sarvam_translate/chrf.svg)); scripts:
`scripts/eval_translate.py` and `scripts/eval_translate_report.py`.

## Method

- **Data:** IN22-Gen (`ai4bharat/IN22-Gen`, revision `e042ab3d`, parquet SHA-256 `e79c7889...`; evaluation only, never
  calibration). 50 of its 1,024 sentences, the same 50 ids in every language and direction, drawn by a seeded order that
  spreads domain and source evenly (seed 0). Every language in IN22's own script: Manipuri in Meitei Mayek, Santali in
  Ol Chiki, Sindhi in Devanagari, Kashmiri in Perso-Arabic. 44 directions x 50 = 2,200 translations per config.
- **Configs:** `sarvam-int4` (the calibrated policy, `model.int4.qt`, 2.99 GB), `aya-int4` (2.34 GB), `qwen-int4`
  (1.41 GB). `sarvam-int8` (4.12 GB) was scored for bits per character only; its generation, run last as optional, was
  stopped. Deltas and CIs are against `sarvam-int4`.
- **Prompts**, each through the model's own chat template: Sarvam the card's (system "Translate the text below to
  {Target}.", the text as the user turn); Tiny Aya and Qwen3.5 a neutral one that names both languages ("Translate the
  following text from {Source} to {Target}. Reply with the translation only.").
- **Decoding:** greedy, batch 8 in a closed loop, at most 32 + 4 x (source characters) new tokens.
- **chrF++:** sacrebleu `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`, corpus level, on IN22's own
  protocol (IndicTrans2 `compute_metrics.sh`): Indic targets through IndicNLP normalize (default options) +
  trivial_tokenize (indic-nlp-library 0.92, urduhack 1.1.1), English raw.
- **CIs:** paired bootstrap, 95%, 1000 resamples; a macro row resamples the shared ids jointly over its languages, so
  the CI carries the correlation between languages that translate the same sentence.
- **Bits per character:** teacher-forced, each model's log-probability of the human reference given its prompt, in
  bits, over the reference's characters (so models with different tokenizers compare); without and with the model's
  end-of-answer token.
- **Macro rows:** all 22 languages; Tiny Aya's 9 (the languages its card claims: Bengali, Gujarati, Hindi, Marathi,
  Nepali, Punjabi, Tamil, Telugu, Urdu); one script (18: without Manipuri, Sindhi, Kashmiri and Santali, which are
  written in more than one).

## chrF++, English -> X (higher is better)

In brackets: the difference from Sarvam INT4 and its 95% CI.

| language | Sarvam INT4 | Tiny Aya INT4 | Qwen3.5-2B INT4 |
|---|---:|---:|---:|
| Assamese | 48.3 | 20.9 (-27.4 [-31.7, -23.3]) | 14.0 (-34.3 [-38.0, -30.6]) |
| Bengali | 54.5 | 44.9 (-9.6 [-12.2, -7.2]) | 30.2 (-24.3 [-29.4, -19.4]) |
| Bodo | 49.8 | 1.5 (-48.3 [-52.6, -44.5]) | 2.0 (-47.9 [-51.7, -44.0]) |
| Dogri | 59.5 | 2.7 (-56.8 [-60.0, -53.4]) | 8.3 (-51.2 [-55.1, -46.8]) |
| Konkani | 46.2 | 15.5 (-30.6 [-35.1, -25.2]) | 3.9 (-42.2 [-45.3, -38.9]) |
| Gujarati | 54.3 | 43.5 (-10.8 [-13.3, -8.8]) | 17.3 (-37.0 [-41.4, -31.8]) |
| Hindi | 55.6 | 50.3 (-5.4 [-8.2, -2.6]) | 32.6 (-23.1 [-29.9, -16.4]) |
| Kannada | 55.2 | 38.3 (-16.9 [-19.8, -14.1]) | 17.7 (-37.6 [-42.0, -32.4]) |
| Kashmiri | 42.8 | 18.5 (-24.2 [-28.0, -20.9]) | 3.8 (-39.0 [-42.3, -35.9]) |
| Maithili | 50.4 | 23.9 (-26.5 [-31.8, -22.2]) | 19.3 (-31.1 [-34.4, -27.4]) |
| Malayalam | 54.8 | 32.0 (-22.8 [-28.1, -17.8]) | 15.7 (-39.2 [-43.3, -34.7]) |
| Marathi | 53.4 | 42.8 (-10.6 [-15.4, -7.0]) | 24.4 (-29.1 [-33.1, -24.9]) |
| Manipuri | 50.0 | 1.0 (-49.0 [-52.4, -45.7]) | 1.0 (-49.0 [-52.5, -45.7]) |
| Nepali | 45.7 | 44.7 (-1.0 [-9.4, +9.3]) | 23.6 (-22.1 [-32.7, -9.5]) |
| Odia | 50.6 | 13.9 (-36.6 [-39.8, -33.4]) | 9.1 (-41.5 [-44.9, -38.3]) |
| Punjabi | 46.7 | 42.8 (-3.9 [-6.3, -1.6]) | 18.0 (-28.7 [-34.5, -22.6]) |
| Sanskrit | 41.6 | 18.2 (-23.4 [-27.1, -19.8]) | 12.1 (-29.5 [-32.5, -26.7]) |
| Santali | 39.7 | 0.4 (-39.2 [-42.8, -35.3]) | 0.6 (-39.1 [-42.5, -35.4]) |
| Sindhi | 41.7 | 1.2 (-40.5 [-44.4, -36.5]) | 13.6 (-28.1 [-31.5, -24.4]) |
| Tamil | 52.5 | 38.1 (-14.4 [-21.1, -8.8]) | 23.4 (-29.1 [-33.3, -25.0]) |
| Telugu | 51.6 | 42.2 (-9.4 [-12.4, -6.5]) | 19.5 (-32.1 [-36.8, -27.0]) |
| Urdu | 64.8 | 50.7 (-14.1 [-19.7, -9.5]) | 29.8 (-34.9 [-40.4, -30.5]) |
| **macro, all (22)** | **50.4** | 26.7 (-23.7 [-25.2, -22.1]) | 15.4 (-35.0 [-36.4, -33.4]) |
| **macro, Tiny Aya's (9)** | **53.3** | 44.4 (-8.8 [-11.1, -6.4]) | 24.3 (-28.9 [-31.8, -25.4]) |
| **macro, one script (18)** | **52.0** | 31.5 (-20.5 [-22.0, -18.9]) | 17.8 (-34.2 [-35.7, -32.4]) |

## chrF++, X -> English (higher is better)

| language | Sarvam INT4 | Tiny Aya INT4 | Qwen3.5-2B INT4 |
|---|---:|---:|---:|
| Assamese | 45.1 | 34.4 (-10.7 [-23.6, +4.3]) | 43.8 (-1.3 [-12.7, +10.7]) |
| Bengali | 55.0 | 55.6 (+0.6 [-3.0, +4.4]) | 48.3 (-6.7 [-11.3, -2.1]) |
| Bodo | 50.9 | 19.7 (-31.2 [-35.8, -25.8]) | 11.8 (-39.1 [-42.5, -34.4]) |
| Dogri | 56.3 | 39.6 (-16.7 [-23.3, -10.2]) | 27.5 (-28.8 [-34.3, -22.2]) |
| Konkani | 50.5 | 31.3 (-19.3 [-26.0, -12.1]) | 32.5 (-18.1 [-24.8, -10.3]) |
| Gujarati | 55.9 | 55.5 (-0.4 [-6.3, +5.3]) | 41.9 (-14.0 [-20.9, -6.1]) |
| Hindi | 54.7 | **57.8 (+3.1 [+0.4, +6.1])** | 49.3 (-5.4 [-8.4, -2.0]) |
| Kannada | 53.0 | 41.5 (-11.5 [-16.0, -7.5]) | 34.2 (-18.8 [-27.9, -6.5]) |
| Kashmiri | 43.7 | 20.5 (-23.3 [-40.1, -4.3]) | 16.4 (-27.3 [-39.5, -14.2]) |
| Maithili | 54.6 | 45.2 (-9.4 [-13.3, -5.8]) | 30.2 (-24.4 [-37.7, -10.2]) |
| Malayalam | 54.3 | 32.5 (-21.8 [-29.5, -13.2]) | 38.1 (-16.2 [-24.0, -7.9]) |
| Marathi | 54.0 | **58.7 (+4.7 [+1.3, +8.5])** | 48.1 (-5.9 [-8.6, -2.6]) |
| Manipuri | 50.2 | 11.1 (-39.1 [-44.1, -33.5]) | 4.6 (-45.6 [-50.5, -40.6]) |
| Nepali | 60.1 | 59.9 (-0.2 [-4.3, +3.9]) | 50.9 (-9.2 [-12.7, -5.6]) |
| Odia | 55.3 | 24.7 (-30.6 [-36.7, -24.5]) | 43.5 (-11.8 [-16.3, -7.3]) |
| Punjabi | 50.0 | 52.9 (+2.9 [-1.1, +6.8]) | 38.5 (-11.5 [-17.4, -5.5]) |
| Sanskrit | 50.0 | 33.4 (-16.6 [-23.4, -9.1]) | 24.4 (-25.6 [-32.3, -18.3]) |
| Santali | 42.7 | 17.2 (-25.5 [-30.7, -21.0]) | 9.0 (-33.6 [-38.5, -28.6]) |
| Sindhi | 47.3 | 28.9 (-18.4 [-24.3, -11.6]) | 19.2 (-28.1 [-32.6, -22.3]) |
| Tamil | 53.1 | 51.7 (-1.4 [-5.4, +2.3]) | 36.1 (-17.0 [-25.9, -7.0]) |
| Telugu | 53.9 | 56.7 (+2.8 [-1.9, +7.0]) | 45.5 (-8.4 [-11.9, -4.7]) |
| Urdu | 57.9 | **61.8 (+3.9 [+0.0, +7.6])** | 50.0 (-7.9 [-11.0, -4.4]) |
| **macro, all (22)** | **52.2** | 40.5 (-11.7 [-14.4, -9.0]) | 33.8 (-18.4 [-21.2, -15.1]) |
| **macro, Tiny Aya's (9)** | 54.9 | **56.7 (+1.8 [-0.9, +4.2])** | 45.4 (-9.5 [-12.7, -6.0]) |
| **macro, one script (18)** | **53.6** | 45.2 (-8.4 [-11.1, -5.6]) | 38.6 (-15.0 [-18.6, -11.0]) |

## Reading the results

- **Sarvam leads into every language.** English -> X, it is ahead of both general models in all 22 (macro 50.4 vs 26.7
  and 15.4), every CI below zero but one: Tiny Aya on Nepali (-1.0, CI [-9.4, +9.3]); next closest is Punjabi (-3.9). In Tiny Aya's own 9
  languages the gap is still 8.8 points (53.3 vs 44.4). Out of English is where the general models fall furthest.
- **The rare scripts are Sarvam's alone.** Into Manipuri (Meitei Mayek), Santali (Ol Chiki), Bodo and Dogri, Tiny Aya
  and Qwen3.5 score 0.4-8.3 against Sarvam's 39.7-59.5, and almost none of their outputs is in the target's script
  (under 50% of letters: Tiny Aya 50/50 for each of the four and Sindhi; Qwen3.5 50/50 Manipuri and Santali, 49/50
  Bodo). What they write instead, counted by each output's majority script: Tiny Aya's Sindhi 47/50 Perso-Arabic (IN22's
  Sindhi is Devanagari), its Dogri 31/50 Gurmukhi, its Bodo Kannada, Georgian or Latin; Qwen3.5's Manipuri 45/50 and
  Bodo 47/50 Bengali script. For Sindhi and Manipuri, both also written in the script the general model chose, part of
  the zero is the script, not the translation; chrF++ cannot tell those apart.
- **Into English, Tiny Aya beats Sarvam in Hindi, Marathi and Urdu.** +3.1 [+0.4, +6.1], +4.7 [+1.3, +8.5] and
  +3.9 [+0.0, +7.6] (Urdu's CI touches zero); Punjabi +2.9 and Telugu +2.8 lean the same way inside their CIs; over its
  9 languages it is +1.8 [-0.9, +4.2], level. Outside them Sarvam leads by 9-39 points. Qwen3.5 is behind Sarvam in all
  44 directions, every CI below zero but Assamese -> English (-1.3 [-12.7, +10.7]).
- **Into English is easier for everyone**, and the spread is smaller: macro Sarvam 52.2, Tiny Aya 40.5, Qwen3.5 33.8.
  Reading a language takes less than writing it.

## Bits per character

Each model's cost of the human reference (lower is better), without the end-of-answer token. INT8 vs INT4 is Sarvam's
quantization; the other columns compare models.

| direction, rows | Sarvam INT8 | Sarvam INT4 | Tiny Aya INT4 | Qwen3.5-2B INT4 |
|---|---:|---:|---:|---:|
| English -> X, all 22 | 0.632 (-0.004 [-0.005, -0.004]) | 0.636 | 2.281 | 2.008 |
| English -> X, Tiny Aya's 9 | 0.526 | 0.528 | 0.764 | 1.180 |
| English -> X, one script (18) | 0.558 | 0.561 | 1.516 | 1.612 |
| X -> English, all 22 | 0.526 (-0.003 [-0.009, +0.003]) | 0.529 | 0.569 (+0.040 [-0.002, +0.084]) | 0.621 (+0.092 [+0.052, +0.132]) |
| X -> English, Tiny Aya's 9 | 0.512 | 0.515 | **0.379 (-0.137 [-0.185, -0.091])** | 0.491 (-0.024 [-0.069, +0.016]) |
| X -> English, one script (18) | 0.515 | 0.518 | 0.494 (-0.024 [-0.069, +0.020]) | 0.553 (+0.034 [-0.006, +0.077]) |

- **INT8 vs INT4 is small in bits.** The calibrated INT4 costs 0.004 bits per character into the Indic languages (CI
  [-0.005, -0.004]: measurable) and 0.003 into English (CI across zero). Its largest per-language costs are Kashmiri
  (0.012), Sanskrit, Santali and Sindhi (0.010 each), the same end as the calibration's held-out KL, where Kashmiri and
  Sindhi were among the five worst languages for flat INT4.
- **Into the Indic languages, bits agree with chrF++:** Tiny Aya's cost of the reference is 3.6x Sarvam's over all 22
  and 13x on Manipuri (9.731 vs 0.744), where it barely knows the script.
- **Into English they do not.** Tiny Aya's reference cost is lower than Sarvam's in all 9 of its languages (Hindi 0.381
  vs 0.528, every CI below zero), and Qwen3.5's is lower in Hindi (0.468 vs 0.528, CI [-0.105, -0.018]) although its
  chrF++ there is 5.4 points behind. The next section looks at why.

### Why Tiny Aya and Qwen3.5 find the English references cheaper than Sarvam does

From the per-sentence records in `build/eval_translate/` (CPU only, no model loaded; X -> English over Tiny Aya's 9
languages, 450 references per config, unless noted):

- **Not the tokenizer.** Bits are per character, and on the English references the two tokenizers spend the same 0.214
  tokens per character.
- **Sarvam is costlier across the whole distribution**, not in a few outliers: median sentence 0.454 bits per character
  vs Tiny Aya's 0.336 (Qwen3.5 0.447); 90th percentile 0.880 vs 0.612; 6.0% of references above 1 bit per character vs
  0.4%. Sarvam is cheaper than Tiny Aya on 21.6% of the sentences.
- **The excess sits where Sarvam's own translation departs from the reference.** Split by the sentence chrF++ of
  Sarvam's greedy output: in the third where it matches the reference worst (chrF++ 12-46) Sarvam pays 0.236 bits per
  character more than Tiny Aya; in the middle third (46-61) 0.145; in the best third (61-100) 0.043. Over those 450
  sentences Sarvam's bits per character and its own chrF++ correlate at -0.70.
- **A few references cost Sarvam almost nothing.** 30 of its 1,100 X -> English references score under 0.05 bits per
  character, and all 30 are 4 sentences seen from many languages (one, about Bristol's rugby and cricket clubs, in 19
  of 22); its greedy output equals the reference word for word in 9 of those 30. Tiny Aya and Qwen3.5 have none under
  0.05, and Sarvam has none into the Indic languages.
- **The opposite direction is normal:** into the Indic languages Sarvam's cost is the lowest in every language.

What this suggests (speculation, not tested): the translation fine-tune concentrated Sarvam's English output on its
own phrasing, so when the human reference is worded differently each of its words is more surprising to Sarvam than to
a general model that spreads its probability over many English phrasings. chrF++ scores only the greedy output, and
there Sarvam's phrasing is as good or better; bits per character score the whole distribution on someone else's
wording. The 4 near-free sentences are what having seen them, or a close paraphrase, would look like; IN22 is
AI4Bharat's and the model was built with AI4Bharat, but nothing here shows what its training data held. A direct test
(TBD): the entropy of each model's next-token distribution along its own greedy output, and Sarvam's cost of Tiny
Aya's outputs.

**The tables with the end-of-answer token are not comparable for Sarvam.** With it, Sarvam's X -> English cost rises
from 0.529 to 0.784 while Tiny Aya's stays at 0.570: the median end token costs Sarvam 27 nats, and Tiny Aya 0.02 and
Qwen3.5 0.007. The harness scores `<end_of_turn>` (106) as Sarvam's end of answer, but Sarvam ends a translation with
`<eos>` (1): all 7 stops in the M3 answer key are id 1. Both are in its stop list, so generation is unaffected; the
end-token tables in `raw/sarvam_translate/tables.md` undercount Sarvam until it is scored with id 1 (TBD).

## Tokens per character

The scored references through each model's tokenizer (no BOS). Fewer tokens per character means fewer decode steps
for the same translation.

| language | characters per sentence | Sarvam-Translate | Tiny Aya | Qwen3.5 |
|---|---:|---:|---:|---:|
| English | 155 | 0.214 | 0.214 | 0.221 |
| Hindi | 156 | 0.281 | 0.304 | 0.432 |
| Bengali | 143 | 0.280 | 0.390 | 0.510 |
| Tamil | 183 | 0.282 | 0.303 | 0.488 |
| Marathi | 153 | 0.302 | 0.322 | 0.500 |
| Nepali | 148 | 0.325 | 0.330 | 0.505 |
| Urdu | 148 | 0.335 | 0.375 | 0.431 |
| Maithili | 146 | 0.347 | 0.387 | 0.550 |
| Malayalam | 176 | 0.363 | 1.656 | 0.594 |
| Dogri | 151 | 0.374 | 0.417 | 0.528 |
| Telugu | 152 | 0.390 | 0.411 | 0.620 |
| Kannada | 163 | 0.408 | 1.958 | 0.652 |
| Sanskrit | 150 | 0.408 | 0.439 | 0.589 |
| Konkani | 145 | 0.410 | 0.410 | 0.541 |
| Assamese | 153 | 0.411 | 0.610 | 0.649 |
| Sindhi | 152 | 0.432 | 0.455 | 0.565 |
| Gujarati | 142 | 0.433 | 0.442 | 0.695 |
| Bodo | 160 | 0.457 | 0.498 | 0.589 |
| Punjabi | 140 | 0.569 | 0.481 | 0.797 |
| Kashmiri | 157 | 0.570 | 0.616 | 0.697 |
| Odia | 167 | 0.730 | 1.554 | 0.829 |
| Santali | 162 | 1.016 | 2.494 | 2.511 |
| Manipuri | 149 | 1.830 | 1.824 | 2.508 |

- **Meitei Mayek is spelled in bytes.** 68.4% of the Manipuri references' Sarvam tokens are byte-fallback tokens (1.25
  per character; measured with the engine's tokenizer on the same 50 sentences). At 1.830 tokens per character a
  Manipuri translation takes 6.5x the decode steps of a Hindi one of the same length (0.281), so at the same tok/s it is
  served at about a sixth of the characters per second; that is the tokenizer, not the engine.
- **Ol Chiki has its letters but few merges:** 1.016 tokens per character with only 3% byte tokens. Tiny Aya and
  Qwen3.5 spend 2.5 on it.
- Sarvam's tokenizer is the most economical on 19 of the 22 languages, level with Tiny Aya's on Konkani; Tiny Aya's
  is cheaper on Punjabi (0.481 vs 0.569) and, by 0.3%, on Manipuri;
  Tiny Aya's is 4-5x Sarvam's on Kannada and Malayalam and 2x on Odia.

## Speed and memory

Generation as the harness ran it: greedy, a closed loop of 8 requests in flight.

| config | weights (GB) | output tok/s | output chars/s | TTFT p50 / p90 (s) | peak GPU (GiB) | peak footprint (GiB) | requests timed | preempted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sarvam INT4 | 2.99 | 21.0 | 59 | 1.02 / 1.83 | 4.52 | 5.57 | 2,136 | 1 |
| Tiny Aya INT4 | 2.34 | 19.6 | 85 | 1.04 / 4.81 | 3.06 | 4.74 | 715 | 18 |
| Qwen3.5-2B INT4 | 1.41 | 41.4 | 170 | 0.57 / 5.59 | 2.44 | 3.84 | 465 | 29 |

**Caveat: each row covers only the run segments that recorded speed, which differ by config.** A generation that
failed and resumed wrote speed only for the segment that finished: Tiny Aya's row is its last 715 of 2,200 requests and
Qwen3.5's its last 465, and both of those segments were X -> English only (English output, more characters per token:
hence their characters per second); Sarvam's row is 2,136 of its 2,200 over two segments, half of them into the Indic
languages, with pools of 384 and then 192 blocks (the others 384). TTFT is over all 2,200 records. So the rows compare
neither the same work nor the same conditions; for engine speed use `compare.md` and the M4 profile (Sarvam INT4
24.1 tok/s single stream in a fresh process).

## Threats to validity

- **50 sentences per direction**, the same 50 ids in every language. The CIs cover the choice of sentences, not of
  test set or domain.
- **One reference per sentence**, and chrF++ is character n-grams: a correct translation in another script (Sindhi,
  Manipuri above) or with other wording scores low. Marchisio et al. (2024, arXiv 2407.03211) found that automatic
  metrics understate quantization damage in non-Latin scripts; no human judged these outputs.
- **Different prompts.** Sarvam gets its card's prompt; the general models a neutral one that names the source. The
  harness has a `--prompt` control (another prompt than the model's own); it was not run (TBD).
- **Every config is 4-bit.** No full-precision translation exists for any model (bf16 does not fit in 8 GB), so
  quantization and model are confounded across models; within Sarvam, INT8 vs INT4 is known only in bits per character
  (its INT8 generation was stopped), so its INT8-vs-INT4 chrF++ is TBD.
- **Possible overlap with training data.** IN22 is AI4Bharat's and Sarvam-Translate was built with AI4Bharat. The 4
  sentences Sarvam reads back almost for free (above) fit that, but the data cannot show it.
- **Sarvam INT4's translations come from two code versions** (`6f1a79b-dirty` and `a9cbe98-dirty`, recorded per
  segment); the weights file is the same (header SHA-256 `e604c9d9...`).
- **Bits per character with the end token** use `<end_of_turn>` for Sarvam, which ends with `<eos>` (above).
- **Qwen3.5's tokenizer applies NFC:** 191 of its 2,200 scored references decode to other text, so its bits are of the
  NFC text over the reference's own characters.
- **One fanless M2**, thermals uncontrolled over a run of more than a day; speed rows as cautioned above.

## Reproduce

```bash
scripts/.venv/bin/python scripts/eval_translate.py data                     # IN22-Gen (gated: accept its terms) -> models/eval/in22-gen/
scripts/.venv/bin/python scripts/eval_translate.py generate sarvam-int4     # per config; resumes where a run stopped
scripts/.venv/bin/python scripts/eval_translate.py score sarvam-int4 --n 50 # bits per character of the references
scripts/.venv/bin/python scripts/eval_translate.py report --baseline sarvam-int4
```

The same `generate` and `score` for `aya-int4`, `qwen-int4` and `sarvam-int8`; `report` writes `tables.md`,
`numbers.json`, `numbers.csv` and `chrf.svg`. Sarvam INT4 took ~93 min to generate its 2,128-request segment and
~32 min to score; Sarvam INT8 ~3.4 h to score.
