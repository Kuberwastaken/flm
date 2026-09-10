# BabyLM evaluation implementation declaration

This supplements `BABYLM-PROTOCOL.md` without changing the running training
protocol. Declared before any BabyLM model test likelihoods or continuations
were inspected. The twelve registered runs must finish before the test command
freezes all validation-selected checkpoint hashes and reads model test losses.

## Complete likelihoods

Score every target in the complete official test blocks, excluding inserted
boundary IDs 0 and 1. Start each block at zero state with its BOS input, and carry
each architecture's native state through the entire block. The Transformer keeps
its declared 96-position attention window. Use chunks of 96 input positions and
batches of eight blocks, sorted by length for efficiency. Right padding occurs
only after a block ends and never contributes a target or affects another row.
Accumulate token losses in float64 after float32 model inference. Verify this
batched path against separate-block scoring for all three architectures.

Save completed batches atomically. Resume only with the identical checkpoint,
tokenizer, test-cache manifest, evaluator source hashes and batching settings.
This is a computational optimization, not truncated evaluation. Validate every
block's scored text-token count and decoded byte total against the prepared cache.

Report pooled bits per UTF-8 byte and same-tokenizer perplexity, then the same
quantities separately for all six corpus components. Apply the previously fixed
whole-block overlap flags to these same scores for the secondary analysis; no
second pass, text splicing or new threshold is needed. Preserve the official
mixture for the primary result and report the bytes/blocks excluded per component.

Within each data size and training seed, compare FLM with GRU and Transformer
using 10,000 paired block-bootstrap draws, generator seed 31415. Resample blocks
within each component, retaining its number of blocks. Use the identical draws
for each model comparison. This conditions on the observed component mixture
and trained checkpoints. Artificial blocks can share unknown documents; these
intervals are not independent-document intervals or a replacement for more seeds.
Report both training seeds, their mean and sample standard deviation. Do not
combine the official and overlap-filtered measurements into one score.

## Unedited generation

`data/prompts/babylm-original.json` fixes twelve original prompts, grouped into
conversation, narrative, explanation and procedural text. These group labels
describe the prompts, not established model abilities or corpus partitions.
For every frozen checkpoint, generate every prompt with sampling seeds 17 and 29,
temperature .8, top-k 40 and a maximum of 256 new tokens. This yields 288
continuations across the twelve models. Each prompt resets state and its sampling
generator. Stop on EOS; exclude BOS and tokens containing ASCII control bytes
other than newline/tab. This display filter is recorded and is absent from
likelihood evaluation. Preserve generated token IDs, decoded text, EOS/length
termination and any invalid UTF-8 decoding. Publish failures without regeneration
or editorial correction.

Report descriptive repetition statistics (repeated token 4-gram fraction,
distinct token bigram fraction, longest identical-token run). They describe
degeneration, not truth, coherence, instruction following or a semantic quality
ranking. Grammar and learned physical behavior remain separate evaluations.

## Reproduction

After the registered queue finishes, run `python -m flm.babylm_test`, then
`python -m flm.babylm_samples`. Both commands fail before scoring/sampling if
any registered run is incomplete or its input/checkpoint identity has changed.
Intermediate batch caches stay in ignored `runs/babylm-evaluation`; final
measurements and continuations go to `reports/babylm`. Raw corpus text and model
checkpoints are not redistributed by these commands.

The official challenge is described at <https://babylm.github.io/>. This is our
declared compact comparison, not an official challenge submission or a claim
that our scores use every component of its external evaluation pipeline.
