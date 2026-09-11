# Coordinating the eight-condition learning-rule experiment

The [coordinator](../flm/language_learning_study.py) connects the existing
[training inputs](LANGUAGE-LEARNING-INPUTS.md), [cost pilot](LANGUAGE-LEARNING-TIMING.md),
[resumable trainer](LANGUAGE-LEARNING-RUNNER.md) and
[per-run validation selector](LANGUAGE-LEARNING-VALIDATION.md). It registers four
learning rules crossed with seeds 42 and 43, fits them serially, and freezes all
eight validation selections together before the held-out scorer may run.

**The coordinator is implemented; the official study has not been initialized.**
The [dated readiness record](../reports/language-eligibility/study-preparation.json)
finds neither completed pilot costs nor an official protocol, identity or
selection file. The priority BabyLM queue is unfinished. No official training
budget has been chosen. The [held-out scorer](LANGUAGE-LEARNING-TEST.md) is
implemented and tested on artificial fixtures, but has not run on this official study.

## Freeze the experiment before fitting

Initialization requires all twelve priority BabyLM completion records and their
selected checkpoint hashes, a complete eight-condition cost pilot, an explicit
settings JSON and a nonempty `docs/LANGUAGE-LEARNING-PROTOCOL.md`. The latter is
the future experiment's protocol; this preparation note is not a substitute for
it. No default training budget is supplied by the coordinator.

The pilot gate checks all declared source hashes, condition identities, fifteen
update observations per condition, warmup flags, positive timing values, summed
exposure and matching sampled text within each seed. It requires the full-window
pilot settings. Initialization additionally replays the pilot sampler on the
verified training inputs and checks its token digest and measured token/byte
counts. The chosen study's batch, sequence, context warmup, thread count and
boundary mask must match those measured dimensions. Its total updates, learning
rate schedule and checkpoint cadence remain explicit choices made after costs.

Before any condition directory exists, the coordinator writes a deterministic
study identity binding:

- All eight conditions, their initial tensor hashes, settings, trainable counts
  and frozen parameter names. Initial tensors and training documents must match
  across rules within each seed.
- Exact training-source, tokenizer and graph bindings, plus the fixed validation
  panel and its scoring denominators.
- Prepared test-cache metadata and the tokenization card, read without opening
  test-token payloads. The scorer verifies those bound payload hashes after selection.
- The protocol and measured cost report, twenty-one source modules, the complete evaluation policy, Torch and
  NumPy versions, and the earliest-exact-minimum validation-selection rule.

An existing identity cannot be changed in place. Initialization refuses if a
condition directory already exists without an identity, preventing a retrospective
freeze of this registered directory. This is a local reproducibility mechanism,
not an external preregistration service or proof of an experimenter's prior access.

## Train and resume the entire inventory

The `train` operation verifies the frozen context, then visits BPTT, fixed-core,
eligibility and no-history conditions for seed 42 and then seed 43. Each run gets
a fresh initial model and a binding to the frozen study file. The existing
trainer resumes only from committed, verified checkpoint records; it does not
initialize a new condition from pilot weights or another condition's fitted model.

The coordinator holds a study-wide writer lock. The trainer also holds its
per-condition lock. Code and input context are rechecked between conditions and
after each fit. A failure stops the queue and retains completed checkpoints for
resumption. All conditions retain the same declared update budget and masks;
there is no performance-based early stopping or dropping of unsuccessful runs.

## Freeze all selections before test access

The `select` operation first requires completion records for every registered
condition. It then verifies the frozen context and invokes the existing per-run
selector for each complete run. That selector restores every training payload,
checks optimizer/sampler/exposure history, and computes or verifies every saved
checkpoint's validation record. Cached validation likelihoods are not independently
rerun; their identities and arithmetic are checked as described in the selector note.

The coordinator requires identical final token and byte exposure across learning
rules within each seed. It records all eight selected checkpoint hashes, update
numbers and validation BPB, along with completion and per-run selection checksums.
Before writing the study selection, it rechecks source files and every selected
record and checkpoint. A later change causes selection verification to fail.

This operation reads no test-token payloads and produces no test scores. The
held-out scorer calls this whole-inventory gate and verifies the bound test
metadata before reading payloads. Completing the actual fits, running the scorer
and publishing all paired results remain required work.

## Verification and eventual commands

Eight tests exercise the full initialize/train/select sequence for eight tiny
artificial conditions, with test-token files deliberately absent. Other checks
cover immutable and complete identities, refusal to freeze after condition
directories exist, refusal of incomplete runs before validation loading,
changed source/protocol/test metadata, damaged selected checkpoints, invalid
pilot dimensions and exposure arithmetic, and missing priority/cost evidence.
These synthetic fits are software tests, not BabyLM learning results.

```powershell
python -m unittest discover -s tests -p test_language_learning_study.py -v
python -m scripts.language_learning_study_preflight --output reports/language-eligibility/another-dated-study-preparation.json
# After actual costs, an explicit protocol/settings file, and exited priority jobs:
python -m flm.language_learning_study initialize --settings work/language-learning-settings.json
python -m flm.language_learning_study train
python -m flm.language_learning_study select
```

Actual process handles must be checked before launch: completion records do not
prove that another trainer has exited. This study changes learning rules on the
original ranked subset. It does not replace the separate neuron-selection,
BabyLM baseline, sensory or physical experiments.
