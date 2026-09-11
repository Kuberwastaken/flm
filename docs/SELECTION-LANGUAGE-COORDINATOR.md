# A complete-group coordinator for the selection experiment

The [coordinator](../flm/selection_language_study.py) can freeze an explicit
selection matrix, fit it serially with resume, and select validation checkpoints
only after every registered condition completes. **No official matrix, budget,
protocol, cost result or selected-graph language fit exists yet.** The priority
BabyLM baseline queue is still running. An official held-out scorer and its
inference policy also remain to be implemented before registering the experiment.

This is preparation for the reviewer's selection question. It does not establish
that a circuit-based subset helps language learning. The anatomical candidates
remain truncated operational selections with substantial missing boundary input.

## Choose groups, not favorable individual controls

The input accepts candidate-group names, not a list of individual fits. Each
chosen group must contain the following complete comparison:

| Selection rule | Node selections | Wiring versions per selection | Training seeds | Fits |
|---|---:|---:|---:|---:|
| Operational KC-centered candidate | 1 | Measured + three rewires | 42, 43 | 8 |
| Contact-ranked at the same neuron count | 1 | Measured + three rewires | 42, 43 | 8 |
| Uniform random at the same neuron count | 3 | Measured + three rewires | 42, 43 | 24 |
| Random matched on class, side and fast sign | 3 | Measured + three rewires | 42, 43 | 24 |
| **Complete group** | **8** | **32 graph definitions** | **2** | **64** |

Node-selection seeds are 201, 203 and 207; graph-rewiring seeds are 101, 103 and
107. These are different sources of variation from training seeds 42 and 43.
All eight groups in the released catalog pass this inventory check. Their 512
possible fits are **not** a chosen 512-fit study. The
[dated preparation record](../reports/selection-language/coordinator-preparation.json)
lists the groups, dimensions, allocations, source hashes and still-closed gates.
It executes no model and opens no corpus payload.

Groups are ordered by name, then training seed, then graph label. Every condition
uses fresh parameters, the same tokenizer, training blocks and update settings.
Learned parameters match across the measured and rewired graphs of each subset
at initialization. Non-edge parameters match across the same-size selectors.
Edge counts and parameter allocations can differ between selection methods;
the coordinator records that difference rather than treating it as controlled.

## Measure costs before fixing the exposure

Initialization requires the existing priority-completion/structural-release gate
and the complete [64-original cost pilot](SELECTION-TIMING-PILOT.md). It verifies
the pilot's graph inventory, randomized order, source files, framework versions,
shared-update settings, positive update times, timing sums and exposure sums.
The acquired corpus verifier independently checks the pinned training sources.
All pilot windows and scored token/byte totals must replay on these inputs.
The original seed-42 model and graph-array hashes must match the pilot.

The explicit request contains exactly:

- `groups`: unique candidate-group names from the released catalog.
- `settings`: every field in the shared training `Settings` record. The template
  seed is 42; the coordinator also creates seed 43. Batch size, sequence length,
  warmup, thread count and boundary mask must match the measured pilot.
- `rationale`: the anatomical and computational reason for the group choice,
  written before fitting. Avoid choosing groups after observing language scores.
- `cost_allowance_multiplier`: a finite factor of at least one for costs outside
  the measured updates and uncertainty in extrapolation.
- `wall_time_budget_seconds`: an explicit planning budget for the entire matrix.

For each condition, the estimate is the original selection's mean measured
update time times the proposed update count. The coordinator adds all conditions,
applies the allowance and refuses a matrix whose estimate exceeds the budget.
The rewires and seed 43 have **unmeasured cost proxies**, explicitly labeled in
the identity. Loading, validation, checkpoint I/O and sustained thermal effects
are unmeasured too. This calculation cannot guarantee wall time or automatically
stop training at a deadline; truncating only slower fits would break equal exposure.

The schedule, update count, checkpoint cadence and group choice have no defaults.
A nonempty `docs/SELECTION-LANGUAGE-PROTOCOL.md` must be written before freezing;
that official file does not yet exist. Training includes inserted boundary targets
after context warmup, consistently with the selection cost pilot. The separate
learning-rule experiment uses a different boundary mask and remains separate.

## Freeze, resume and select as one inventory

The frozen identity binds every condition, graph, initialization, corpus identity,
tokenizer, settings, validation panel, pilot, protocol and numerical source hash.
Only the existing test manifest and tokenization-card metadata are bound; no
test-token file is opened. A fresh freeze refuses existing run files, including
directories for unselected groups. Re-initialization must reproduce the identity.

Training holds a study writer lease and uses the tested
[selection adapter](SELECTION-LANGUAGE-TRAINING.md). Each run's declaration binds
the study identity as well as its own graph and corpus. The common runner restores
the complete checkpoint, optimizer, sampler and exposure record after interruption.
Code, protocol and pilot are checked between fits; the complete input context is
reloaded and verified at the end. Completion files do not prove that unrelated
trainers have exited: inspect the actual process handles before starting this queue.

Selection first reconstructs the complete group inventory and refuses missing
completion files before loading validation. For every condition, the shared
validation selector audits all declared checkpoints and chooses the earliest
exact minimum validation BPB. All conditions must have identical final token and
byte exposure within each training seed. Selected payloads and run records are
checked again before saving the immutable whole-study selection.

There is no `test` command in this coordinator. The eventual scorer must require
this complete selection, freeze its inference and comparison policy before any
official training starts, and report all outcomes. Cross-selector comparisons
address selection under unequal edge allocation. Within-subset rewiring contrasts
address retained topology at the same allocation. Neither is an intact-animal
experiment or general evidence for or against biological wiring.

## Software verification

Seven tests passed. The complete synthetic group uses 64 actual tiny fits, with
an injected interruption during the third run, followed by resume, validation
selection and cached-selection revalidation. Test-token payloads are deliberately
absent. Other checks reject incomplete groups, unknown or repeated group names,
missing priority/cost evidence, inadequate planning budgets, retroactive freezes,
shortened identities, changed pilot windows/source/protocol and damaged selected
checkpoints. Timing-gate tests use clearly fabricated schema fixtures.

These are software tests, not BabyLM results or official cost measurements. The
real catalog preflight performs zero initialization, forward or gradient calls.

```powershell
python -m unittest discover -s tests -p test_selection_language_study.py -v
python -m scripts.selection_language_study_preflight --output reports/selection-language/another-dated-coordinator-preparation.json
# Only after actual costs, the final protocol/evaluator, and exited priority jobs:
python -m flm.selection_language_study initialize --request work/selection-language-request.json
python -m flm.selection_language_study train
python -m flm.selection_language_study select
```

The existing baseline, learning-rule, sensory and body experiments keep their
own identities and protocols. This coordinator does not replace their controls.
