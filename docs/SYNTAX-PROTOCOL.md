# BLiMP linguistic diagnostic

Use the original [BLiMP benchmark](https://github.com/alexwarstadt/blimp), pinned
to revision `3e56b06fcabca9b30822fc66435fca6b1aa40bb1`. Its 67 paradigms each
contain 1,000 expert-grammar-generated acceptable/unacceptable sentence pairs.
This synthetic linguistic diagnostic is evaluation data only; it is never added
to FLM's language training. The authors distribute it under CC BY 4.0.

The first bounded panel contains 100 pairs per paradigm, chosen by ascending
SHA-256 of `UID/pairID`, before reading any model scores. All architectures and
training corpora use these same 6,700 pairs. This is explicitly a 10% diagnostic
panel, not the complete official benchmark. Keep a path to full evaluation.

Restore each validation-selected model without any adaptation. Reset recurrent
state or attention cache at the start of every sentence. Insert BOS as context,
then sum next-token negative log probabilities over the original sentence text,
including its punctuation. Exclude BOS and EOS from the score; do not normalize
by tokens, words or bytes. The model succeeds when the acceptable sentence has
strictly smaller summed negative log likelihood. Exact ties count as incorrect
and are reported separately. Right padding is masked from loss and cannot change
an earlier prediction in the causal models; verify against individual scoring.

Report accuracy by paradigm and linguistic category, overall macro-average over
paradigms, seed variation, score margins and exact checkpoint/tokenizer/data
identities. Publish every pair's scores and identifier. Category differences on
this exploratory panel are hypotheses for the full benchmark, not claims of
general grammatical understanding. Scores from models trained on different
corpora/exposures are labeled separately. Do not select training checkpoints or
learning rates using this diagnostic. The primary checkpoint rule stays the
registered corpus validation loss.

The original repository documents corrections to its published numerical tables
on 16 August 2021. Consult its corrected results if quoting external model numbers;
do not silently mix a 6,700-pair FLM panel with full-benchmark published accuracy.

Reference: Warstadt et al. (2020), [BLiMP: The Benchmark of Linguistic Minimal
Pairs for English](https://aclanthology.org/2020.tacl-1.25/), TACL 8, 377–392.
