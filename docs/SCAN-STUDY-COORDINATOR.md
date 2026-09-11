# Whole-study control for instruction transfer

Kuber Mehta · 11 September 2026. This is implementation readiness, not a
registered downstream budget or a SCAN result. The fixed BabyLM comparison and
the chosen neuron-selection study retain priority. No official instruction fit
or test generation has run.

The [coordinator](../flm/scan_study.py) closes a gap between independently
resumable fits and a complete comparison. It freezes all **36 conditions**:
three official splits × three architectures × two source initializations ×
two seeds. Initial and WikiText-trained sources share the WikiText train-fitted
tokenizer. Each split keeps its own training rows and model state; their training
sets must not be combined. The [data and interpretation note](INSTRUCTION-TRANSFER.md)
explains why this is symbolic instruction composition, not conversational or
physical competence.

## Before fitting

Initialization requires completed BabyLM checkpoint records, the completed
chosen selection comparison, the full eighteen-condition train-only cost pilot,
and an explicit `docs/SCAN-PROTOCOL.md` with a separately supplied settings JSON.
**The official cost pilot, protocol, settings and frozen identity do not yet
exist.** The command does not invent a budget. Its record checks do not establish
that another process has exited; the operator must inspect live process/session
handles before launching a pilot or training job.

The coordinator checks the pilot's source/software identities, original model
states, ordered input bindings, all fifteen observation records, measured time
arithmetic and every update's exact sampled exposure. It reconstructs the sampled
row digest from the verified training data. Only seed 42 was included in the
planned pilot; seed-43 cost remains an explicit proxy. Batch size, CPU threads
and the common context must match the measured dimensions. Timing does not select
an optimizer schedule or establish a transfer effect.

Every source is prepared again before the immutable study identity is written.
The identity records the model class/configuration, initial tensors, tokenizer,
ordered training rows, numerical source hashes, optimizer settings, planned
sampled-row digest and exact terminal exposure for every condition. All six
architecture/initialization conditions within a split and seed must receive the
same sampled rows and token exposure. The added-jump split's deliberate repeated
rows retain their original sampling weight. Language pretraining is additional
exposure, so matching downstream updates does not match lifetime compute.

The identity also binds the already acquired dataset card's test metadata and
the evaluation policy. It does not open test text or targets. A new identity is
refused if condition directories already exist; an existing identity can only
be reused unchanged. Changing the budget after downstream fitting is not a
checkpoint-selection procedure.

## Serial fitting and interruption

One OS-held study lease encloses the serial scheduler; each condition has the
fitter's separate writer lease. Before fitting or resuming, the original source
is prepared again and checked against the frozen declaration. Its binding gains
the immutable study identity hash. The existing fitter restores only committed
checkpoint boundaries and preserves the declared schedule, sampler, optimizer,
training history and exact exposure.

Each attempt retains its completed condition records. Exceptions leave a dated
failure record naming the condition and study identity. Resumption uses the same
inventory and revisits earlier completed conditions through the normal checkpoint
verification path. It does not silently replace failed attempts or treat partial
fits as complete.

## The gate before test generation

`select` first reconstructs all 36 condition labels from code, without trusting
a possibly shortened identity file. Every completion marker must exist before
the payload audit begins. It then restores every fixed terminal checkpoint and
checks learned tensors, fixed graph/pooling buffers, optimizer state, finite
nonnegative second moments, full loss history, sampler RNG, sampled row digest
and exact exposure. It checks all committed checkpoint hashes and rechecks the
selected checkpoint/completion hashes after the whole inventory has been audited.

Only then can one immutable `study-selection.json` be written. No validation or
test accuracy chooses a checkpoint: SCAN has no official validation partition in
these splits, and this comparison uses the declared terminal update.

The bound policy requires greedy generation over the full existing vocabulary,
a 49-token output cap including EOS, a common 96-token context and exact complete
action-sequence success with EOS. Invalid tokens, omissions, extra actions and
cap exhaustion remain errors. The shared [generation/scoring runtime](../flm/scan_runtime.py)
implements those primitives. The [whole-partition evaluator](SCAN-EVALUATION.md)
now integrates the provenance-checked loader, this completion gate, resumable
generation and complete-condition comparisons. Its source is bound by the
pre-training study identity. A passed completion gate alone is not a test result.

## Verification and use

The [twelve lifecycle tests](../tests/test_scan_study.py) fit all 36 small synthetic
conditions for six updates, audit every terminal payload, and reproduce all final
model tensors, training histories, sampled rows and exposure after an injected
interruption. Additional faults cover a shortened inventory, changed source or
protocol, a new post hoc identity, duplicate pilot conditions, forged exposure,
rehashed nonfinite tensors, altered sampler state, negative optimizer second
moments and a completion changed late in the audit. No official SCAN test files
are present in those fixture directories.

```powershell
python -m unittest discover -s tests -p test_scan_study.py -v
# Only after measured costs, a written protocol, completed priority studies,
# verified process exit and evaluation preparation checks:
python -m flm.scan_study initialize --settings path/to/declared-settings.json
python -m flm.scan_study train
python -m flm.scan_study select
```

The [dated preparation record](../reports/scan-runtime/study-preparation.json)
separates synthetic software checks from official study status. It contains no
instruction accuracy or timing claim for the benchmark models.
