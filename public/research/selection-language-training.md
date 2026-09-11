# Language-training inputs for the neuron-selection experiment

The [selection adapter](selection-language/selection_language.py) connects the released graphs
to the common BPTT training runner. It can prepare each of the 64 measured induced
subsets and their 192 artificial rewires, with fresh training seeds 42 and 43.
**These are 512 possible initializations, not a chosen 512-fit experiment.** The
final matrix, common training budget and official selection/evaluation protocol
remain unfrozen. No selected-graph corpus fit has run.

The [dated preparation record](selection-language/preparation.json)
verifies the complete acquired training corpus, graph archive, fresh model
initializations and sampled-window identities. It makes no forward calls or
gradient updates and contains no validation/test payloads or language scores.
It does not replace the still-pending measured cost pilot.

## Keep the language corpus independent of graph choice

The common training verifier now exposes `load_corpus` separately from the
original 1,024-neuron graph. It retains the same pinned six BabyLM `train-10m`
sources, publisher revision checks, SHA-256/Git-blob provenance, train-fitted
4,096-entry tokenizer, complete cache verification and byte-exact block decoding.
All 3,351 blocks cover 54,399,840 UTF-8 bytes once, with 18,591,514 text tokens
plus inserted boundaries.

This graph-independent binding contains no reference-graph identity. Each
selection model instead records its own measured/rewired graph and the released
structural archive's checksum. The original `load_inputs` entry point still
adds the historic reference graph for the separate learning-rule experiment.
The refactor does not change the model or optimizer used by the running BabyLM
baseline queue. Historical preparation reports retain their original source hashes.

## Bind the actual selected graph

The adapter reads the pinned structural archive directly, without extracting
paths from it. It verifies the exact 64-original/192-rewire inventory, both
manifests, every graph checksum and every rewiring-to-original binding. Models
use the existing full FLM forward computation, 96-dimensional embeddings, 128
pools and tied readout, with each graph's own neuron and edge counts.

For each model it checks array inventory, dimensions, integer indices, pool
bounds, finite computational arrays, unique body IDs, parameter groups and total
allocation. Artificial rewires must preserve the original incoming slots,
outgoing degrees, source signs, self-edges and fixed metadata while changing
endpoints. Missing anatomical coordinates remain missing: positions do not enter
the language recurrence and are not silently replaced by invented coordinates.

Fresh initialization records both the complete model-state hash and the learned
parameter hash. Within each measured subset and training seed, all three rewires
must have identical learned parameters at initialization. Their graph-buffer
identities differ. Across selectors at the same neuron count and seed, non-edge
parameters also match; edge-gain vectors have different lengths when edge counts
differ. This is not parameter-count matching between selection methods.

No checkpoint, pilot weight or previously trained language embedding initializes
these models. Graph arrays are copied for each model, and initialization preserves
the caller's Torch random state. Sampler seeds remain independent of graph seeds.

## A common fit/resume path

The adapter calls the existing [resumable training runner](language-learning-runner.md)
with BPTT and an explicit settings object and study binding. It uses the
[selection pilot's](selection-timing-pilot.md) convention: exclude context-warmup
positions but include any inserted boundary targets afterward. The pilot now calls
the exact shared BPTT update, including inventory and finite-parameter checks.
Eight pilot tests include exact final model/optimizer and sampled-window parity
with a saved tiny fit. Pilot times will still exclude loading, checkpoint I/O,
validation and extra bookkeeping; those costs need allowance in the final budget. The separate
learning-rule experiment excludes boundary targets; the two studies must not be
silently pooled as though their objectives were identical.

A saved run binds its selected graph, corpus, seed, initialization, adapter source,
settings and supplied study identity. Resuming under another graph or study
binding fails. The underlying runner retains its checkpoint inventory, optimizer,
sampler and exposure checks. Changing graph selection does not change the learning
rule, and changing a learning rule belongs in its separate experiment.

This is a low-level fitting component. It does not itself register a complete
study, choose an affordable matrix, enforce all-condition completion or select
validation checkpoints. The [complete-group coordinator](selection-language-coordinator.md)
now implements registration, serial resume and whole-inventory validation selection;
its synthetic tests pass. The [held-out evaluator](selection-language-test.md)
now binds the scoring and comparison policy before fitting. Actual groups,
budget, protocol and corpus fits remain pending. There is deliberately no command that launches an unfrozen
corpus experiment through this module.

## Verification and next step

Five adapter tests use artificial graph archives and actual tiny updates. They
check complete archive identity, missing/changed graphs, matched initial learned
parameters, isolated graph buffers, missing coordinates, invalid seed/tokenizer
bindings, identical fitted tensors and training histories after resume, and
agreement with a direct implementation of the pilot's boundary-included update.
Six input-verifier tests pass after the corpus/graph separation, including a
corpus-only load with an unusable reference graph.

The real preparation instantiates all 256 released graph definitions for both
training seeds and checks parameter matching. It separately replays fifteen
sampled windows at batch 16, sequence 96 and warmup 16 for each seed. These are
input and initialization checks, not timing measurements or training fits.

```powershell
python -m unittest discover -s tests -p test_selection_language.py -v
python -m unittest discover -s tests -p test_language_learning_inputs.py -v
python -m scripts.selection_language_preflight --output reports/selection-language/another-dated-preparation.json
```

After the priority BabyLM jobs exit, measure the aligned cost pilot. Use those costs and the anatomical rationale to freeze the selected
candidate/control matrix, common exposure, optimizer schedule, validation rule
and held-out gate. Keep cross-selector comparisons distinct from within-subset
rewiring comparisons. Language results will still concern truncated operational
candidates, not intact biological circuits or animal behavior.
