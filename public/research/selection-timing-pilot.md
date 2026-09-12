# Measuring the cost of a selection experiment

The [selection comparison](circuit-selection.md) needs an affordable declared
training matrix. Matching neuron counts does not match edge counts, trainable
parameters or runtime. The new [pilot runner](selection-pilot/selection_pilot.py) prepares
that cost measurement using the existing FLM implementation and BabyLM's
training data. **The full 64-original timing pilot completed on 12 September
2026.** The [timing record](selection-pilot/timing.json) and
[cost decision](selection-pilot/cost-decision-v1.json) support the
registered bilateral visual-KC study: 128 fits at 3,000 updates, estimated
35.37 update-hours before allowance. This measures cost, not language quality.

## Readiness evidence

The [dated preflight](selection-pilot/shared-update-preflight.json) verifies all
64 full-size original graph configurations through two disposable optimizer
updates each, using a tiny synthetic token fixture: batch 1, three input tokens,
one context-warmup position and one CPU thread. Every graph remains unchanged,
all updates are finite and the disposable parameters change. This extends the
earlier forward-only graph check to backward computation and AdamW updates.
It does not establish that every edge has a useful gradient, that a circuit
retains its biological function or that a model learns language.

Separately, the preflight verifies the existing 10M training cache and its
train-fitted 4,096-entry tokenizer against the registered BabyLM identities.
There are 3,351 source-indexed blocks and 18,598,216 BPE token IDs including block
boundaries. These are token counts, not a revision of the dataset's nominal
word budget. The actual corpus is not passed to the synthetic gradient checks.
No validation or test cache is opened. No model, loss values or fixture throughput
estimates are saved.

Eight unit tests cover exact sampled token/byte exposure, matching windows
between graphs, preserved source arrays and global Torch state, malformed inputs,
nonfinite updates, complete shuffled inventory, input fingerprints, priority
gates, retained failed attempts and refusal to overwrite a completed pilot. The
shared-update test compares a complete tiny pilot with a saved fit: final model
tensors, AdamW moments and steps, parameter groups and sampled-window digest
match exactly. It includes boundary targets after warmup. These are artificial
fixtures, not throughput observations.

## The eventual timing run

The command will measure **all 64 original selections** in one fixed shuffled
order, using PCG64 order seed 519. It does not choose the most convenient graph
from each class, and it does not time the 192 artificial rewires. Costs for those
controls must not be presented as measurements from this pilot.

Each condition uses its released graph configuration: the existing fast/slow
FLM, 96-dimensional embeddings, 128 pools, a tied 4,096-entry readout and the
automatic matrix backend. All present selections use the dense backend under
the existing size rule. Initialization and window sampling use seed 42. Each
selection gets the same sampled training windows; a digest check enforces this.
The same seed across different graph sizes does not imply identical initial
parameter tensors.

For each graph, three warmup updates precede twelve measured updates with batch
16, sequence length 96, and the first 16 positions excluded from the training
loss. As in the existing language trainer, boundary targets after those positions
remain included. This is a timing preparation, not a change to the published
training or evaluation masks. The optimizer is AdamW at a constant pilot learning
rate of 0.002, weight decay 0.01 and gradient clipping at 1.0. Its convergence is
not assessed, and this constant rate does not define the future study schedule.

The twelve measured updates present 18,432 input tokens and supervise 15,360
target positions per graph, plus separately recorded warmup exposure. Timing
includes sampling, pilot token-range guards and the exact
`flm.language_learning_train.update` BPTT call used by the selection adapter. That
call includes parameter-inventory checks, warmup masking, learning-rate handling,
forward/backward computation, finite loss/gradient checks, clipping, AdamW and
finite-parameter checks. Loading, initialization, additional full-state/optimizer
audits, exposure bookkeeping, validation and checkpoint I/O are excluded. Token and byte counts
are exact; median update time and aggregate throughput are short-run observations.
Neither is a peak-memory measurement or a guaranteed full-training wall time.

Every condition starts fresh and discards its parameters and optimizer. Only
timings, exposure counts, identities and finite-update checks are retained.
There are no predictions or language-loss values to use for selecting a circuit.
Completed cases from a failed attempt remain visible; a later attempt starts
fresh and does not silently merge measurements from different sessions.

## Running it without disturbing the priority studies

The official command checks all twelve BabyLM completion records and selected
checkpoint checksums, plus the terminal structural archive's identity. It
refuses to start while either prerequisite is unfinished. Completion records
do not establish that processes have exited: inspect the actual training and
generation process/session handles before launch. An OS lock prevents concurrent
selection timing writers.

```powershell
python -m unittest discover -s tests -p test_selection_pilot.py -v
python -m scripts.selection_pilot_preflight --output reports/selection-pilot/another-dated-preflight.json
# Only after the priority jobs have exited and prerequisites are satisfied:
python -m flm.selection_pilot
```

The [preflight implementation](selection-pilot/selection_pilot_preflight.py) keeps
the synthetic readiness check separate from actual corpus timing. It refuses
to overwrite a dated report. The official runner writes per-case attempt
records under the ignored runs directory and a completed inventory to
`reports/selection-pilot/timing.json`. The completed record is now preserved there.

After measured costs are available, freeze the selection/rewiring/training-seed
matrix, common exposure, optimizer schedule, validation selection and test gate.
Keep selector contrasts separate from within-subset wiring contrasts. Retain
the anatomical rationale and boundary audit even if a cheaper matrix is chosen.
The current BabyLM, language topology, core-computation, sensory and physical
experiments remain unchanged.

The [selection-language adapter](selection-language-training.md) now supplies a
common resumable fitting path. Both now call the same update function, and the
pilot records its complete constant-rate settings and twelve source-module
checksums. This aligns the update computation; the excluded I/O, validation and
bookkeeping costs still need allowance in an eventual wall-time budget. The
[earlier preflight](selection-pilot/preflight.json) is retained as a
dated record of the prior direct-update implementation.
