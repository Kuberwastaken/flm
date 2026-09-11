# Held-out evaluation for the four learning rules

The [evaluator](language-learning/language_learning_test.py) scores the eight registered
learning-rule conditions: BPTT, fixed core, forward eligibility and no-history
eligibility, each with training seeds 42 and 43. It uses the same original
1,024-neuron graph and prepared BabyLM test partition for every condition.

**The evaluator is implemented, but no official learning-rule test scores exist.**
The [dated preparation record](language-learning/evaluation-preparation.json)
checks source identities and existing cache metadata only. Official costs,
training protocol, budget, study identity, fits and selections remain pending.
The priority BabyLM training queue must finish before the measured cost pilot.
Temporary artificial fits used in software tests are not corpus results.

## What is frozen before fitting

The [study coordinator](language-learning-study.md) now binds twenty-one source
modules, including this evaluator and the existing complete-block scoring code.
It also stores the entire evaluation policy in its pre-fit identity:

- Batch eight independent blocks, with chunks of 96 input positions and four CPU
  threads. Every block starts with empty recurrent state; state carries between
  chunks within that block. Padding follows each block's last real target.
- Score every text target, excluding inserted boundary targets. Evaluation has
  no context warmup exclusion. Training still uses its separately declared warmup.
- Report the complete prepared test partition and the existing overlap-filtered
  subset. Eligibility comes from the acquired cache metadata and is not chosen
  after seeing losses.
- Report all six pairwise learning-rule contrasts for each seed, in both subsets:
  24 comparisons. Use 10,000 paired bootstrap draws within each source component,
  with seed 31415 and unadjusted descriptive 95% intervals.

Changing code or this policy after initialization invalidates the frozen study.
The official protocol and training budget still need to be written after actual
cost measurements. This local identity mechanism is not an external preregistration.
The test partition is shared with the separate BabyLM baseline study; it is not
an independent new benchmark or proof that no experimenter has seen related results.

## The gate and the actual test data

The evaluator first invokes the whole-inventory completion and validation gate.
That gate restores and audits every declared training checkpoint for all eight
conditions, then freezes the earliest minimum-validation selections together.
A missing final condition or damaged selected checkpoint prevents test access,
even if seven other conditions are ready.

Under study and evaluation writer locks, the evaluator reconstructs the frozen
context and checks the selected payload and record hashes. Only then does it
open the test cache. It verifies the bound metadata and all three cache payload
hashes, unique block IDs, boundary positions, vocabulary bounds, text token and
UTF-8 byte denominators, overlap eligibility, source components and full coverage.
The loaded token arrays become read-only owned copies (about 35 MiB for this
prepared test set), releasing the file mappings before scoring and error recovery.

Each selected model is restored with its exact initialization, training declaration,
optimizer inventory and sampler/exposure history. No weights are updated. All
blocks are evaluated; there is no test-driven checkpoint selection or early stopping.
The recorded likelihood is cross-entropy summed over text targets, divided by
UTF-8 bytes and by log(2) for bits per byte. Token perplexity is also reported.

## Recovery and integrity

The existing complete-block evaluator sorts by block length and stable ID, writes
atomic per-batch records, and resumes from whole completed batches. A batch cannot
silently move between checkpoints, cache identities or scoring settings. Cached
likelihoods are checked for identity, coverage and arithmetic; they are **not
independently recomputed** on resume.

Completed per-condition results include every block likelihood and batch-file
checksums. They cannot be overwritten with different results. Before publishing
the all-condition summary, the runner rechecks selected checkpoints, source files,
test payloads, result files and cached batch hashes. An interrupted run retains
its completed batches but produces no all-condition summary. The caller's Torch
random state and thread count are restored after success or failure.

Per-condition scoring seconds support recovery accounting. They are not isolated
throughput measurements and cannot be used as an efficiency comparison.

## Reading the eventual result

Reports include every source component, both subset definitions, each seed's BPB,
the two-seed mean and sample standard deviation. A filtered component with no
remaining blocks is explicitly null. If the entire filtered partition is empty,
its comparisons are explicitly unavailable rather than filled with invented zeros.

Each contrast is first minus second: a negative difference favors the first rule.
Bootstrap intervals describe uncertainty from resampling the retained artificial
blocks, conditional on those fitted checkpoints and source-component counts.
Adjacent blocks may belong to the same unknown document. These intervals do not
measure training-seed uncertainty, and the 24 comparisons have no multiplicity
correction. The complete inventory should be interpreted together.

A language advantage for eligibility would concern this particular approximate
credit-assignment rule and budget. It would not demonstrate that measured topology
helps, that the update is biologically realistic, or that the model can converse.
Those questions retain their separate controls and studies.

## Verification and eventual execution

Eight tests use temporary synthetic caches and actual tiny training updates for
all eight conditions. Batched/chunked likelihoods match independent direct
whole-block calculations for every selected model. The suite also checks cached
resume, interruption recovery, complete inventory gating, changed source/policy,
corrupted test payloads and selected checkpoints, modified completed records, and
preservation of Torch state and threads. No acquired test text is used by the suite.

```powershell
python -m unittest discover -s tests -p test_language_learning_test.py -v
# Only after measured costs, protocol/identity, and all eight completed fits:
python -m flm.language_learning_test
```

Actual process handles must be checked before launching the official evaluator.
The command has no flags for shortening the test corpus, choosing favorable rules,
or changing bootstrap settings after fitting. Generated text samples, browser
exports, topology-selection fits and language-to-body transfer remain separate work.
