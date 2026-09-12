# BabyLM data and comparison protocol

The next language experiment uses the official 2026 English Strict-Small (10M)
and Strict (100M) training partitions. The immutable repositories and all 24
file identities are in `data/sources/babylm-2026.json`. Acquisition measured
exactly 10,000,000 and 100,000,000 whitespace-delimited training words. These
are word budgets, not FLM's byte-pair token counts.

## Text and boundaries

Preserve every source byte, including speaker markers, headings, whitespace and
line endings. Group consecutive whole lines into blocks with a 16,384-byte target,
closing a block before a line would exceed the target. An oversized individual
line stays intact. Never cross a component file. Record file identity, byte
offsets, line ranges and block hashes. This creates computational reset boundaries;
it does not reconstruct books, speakers or conversations. The training files have
no blank document separators, so document- or speaker-disjointness is unknown.

Train one lossless 4,096-ID byte-level BPE vocabulary on **10M training text only**.
Reuse it for both data scales and every architecture. Reserve two IDs for inserted
block boundaries. Encode all four partitions only after fitting the tokenizer.
Test encoding and text identity checks do not involve model losses or selection.
Store little-endian uint16 tokens in memory-mapped files, with offsets and source
metadata, so the larger corpus need not be inflated to int64 in RAM.

## Overlap

The completed audit compares normalized complete lines of at least eight words
and 40 characters. It collapses whitespace and casefolds, without near-duplicate
matching. The 10M training set matches 289 validation and 1,995 test line
occurrences; the 100M set matches 9,083 validation and 7,459 test occurrences.
Short/common phrases and partial-line matches are outside this audit.

Keep official partitions unchanged for the primary benchmark. For a secondary
check, flag every validation/test block containing any substantial normalized
line present in **either** training partition. Exclude that entire block from the
secondary evaluation; do not splice its remaining text. Use the same exclusion
for both training sizes and every model. Report excluded blocks and bytes per
component. This is an overlap-filtered sensitivity analysis, not proof of zero
contamination. Do not use test losses to change this rule or select checkpoints.

## Compact comparison

The first registered run uses the same approximately 600k-parameter FLM, GRU and
two-layer causal transformer configurations as the WikiText experiment. Train
from scratch, using seeds 42 and 43, a shared sampler, batch 16, 96 input tokens
and 16 warmup positions. Each run receives 12,000 updates: 18,432,000 presented
tokens, with 15,360,000 next-token targets after warmup. Record actual source bytes
and target bytes; report exposure as a fraction of each corpus as well as updates.
Equal compute exposure samples a smaller fraction of the 100M set; it is not an
equal-epoch comparison and does not consume the complete 100M corpus once.

Use the existing AdamW schedule: 100 warmup updates, cosine decay from .002 to
.0002, weight decay .01 and gradient-norm clipping at 1. Select checkpoints every
500 updates on a fixed validation panel. Select eight blocks per component by
ascending SHA-256 of their stable IDs; score up to 1,024 targets per block. This
panel includes official text, with overlap flags reported. Freeze its IDs before
training and share it across sizes and architectures. Selection uses the
byte-weighted pooled score. Also report each component separately.

After all registered runs complete, freeze validation-selected checkpoint hashes.
Evaluate the full official test blocks and the fixed overlap-filtered subset,
reporting bits per UTF-8 byte, same-tokenizer perplexity, component-level scores,
seed variation and paired block-bootstrap intervals. Adjacent blocks from the same
unknown document can be dependent; block intervals do not establish independent
document uncertainty. Publish fixed, original prompts and every continuation,
including repetition and malformed text. Dialogue-like training text alone does
not establish instruction-following ability.

Larger capacity, longer exposure, local learning, syntax tasks and embodied
transfer are separately declared experiments. The official roughly 98M-parameter
GPT-2-family BabyLM models can provide external references, with different
tokenization, data exposure and size clearly identified. They are not substitutes
for the matched baseline or weights inside FLM.

## Sources and rights

- [BabyLM 2026 overview](https://babylm.github.io/).
- [Official Strict-Small corpus](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small).
- [Official Strict corpus](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict).
- [Development partition](https://huggingface.co/datasets/BabyLM-community/BabyLM-dev)
  and [test partition](https://huggingface.co/datasets/BabyLM-community/BabyLM-Test).

The training repositories advertise MIT metadata, which does not replace rights
in each underlying component. Raw corpus text remains local and is not published
in this repository or website. Publisher filtering used toxicity/bias classifiers;
that curation does not guarantee unbiased or entirely human-authored text.
