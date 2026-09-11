# A frozen language core behind a fresh sensory interface

Kuber Mehta · 11 September 2026. Software verification for the separate
[food-response experiment](FOOD-RESPONSE-PLAN.md), with **zero food-adaptation
updates and no new physical trials**. The current BabyLM queue and subsequent
complete-group selection comparison retain priority over food-task fitting.

The original 1,024-neuron FLM can now process the declared six food-sensor
channels through a new input adapter and return three action logits through a
new output adapter. Four interfaces have been exported: the reconstructed
initialization and released language-trained checkpoint for each WikiText
training seed, 42 and 43. This makes the paired comparison executable; it does
not demonstrate learned food behavior or a benefit from language pretraining.

## What is reused

The [preparation record](../reports/food-core/preparation.json) binds the released
WikiText archive, selected checkpoints, graph, source code and every export.
Both selected language checkpoints are from step 6,000. Initial cores are
reconstructed with the original constructor, graph and seed; they are not
validation-selected checkpoints. The paired trained checkpoint record identifies
their corresponding language treatment, while `core_state_sha256` identifies
the actual full source-model tensor state for each interface.

The [PyTorch bridge](../flm/food_core.py) deep-copies the source model and freezes
its parameters. The new sensory projection maps six normalized channels to the
96-dimensional embedding space. The original input projection, signed recurrent
operator, fast and slow state updates, pooling and normalization then run
unchanged. A fresh head maps the 256 pooled features to `turn_right`, `straight`
and `turn_left` logits. All four interfaces use exactly the same fresh adapters
from seed 711. These names do not yet specify physical motor commands.

| Parameter or state quantity | Entries per interface |
|---|---:|
| Original language-model parameters | 600,003 |
| Frozen parameters used by sensory inference | 178,019 |
| Unused lexical embedding and output parameters | 421,984 |
| Fresh trainable input adapter | 672 |
| Fresh trainable action adapter | 771 |
| Fast and slow recurrent state | 2,048 |

The Python object retains the unused lexical parameters, frozen. The portable
export omits them and stores the effective recurrent matrix plus the constants
needed for inference. Exported array sizes are therefore not a count of learned
parameters. State occupies 8,192 bytes in float32, excluding weights and working
memory. A dense 1,024-square recurrent matrix is used in this NumPy reference;
this is not a scaling or throughput result.

## What was checked

Before replacing lexical inputs, the bridge reproduces the native FLM logits
and final recurrent states exactly on the fixed lexical probe recorded by the
[preparer](../scripts/prepare_food_core.py). It leaves the source tensors unchanged.
Unit checks cover state carry/reset, identical adapters without changing the
source RNG, and a finite-difference gradient through the frozen core to the
sensory adapter. A synthetic optimizer step changes only adapter parameters.

Each exported interface was then compared with native PyTorch inference on
466 sensory frames, resetting at the start of each stream:

- 201 frames from each of the completed left and right scripted odor-A trials;
- 64 synthetic frames containing zero, one, isolated-channel and random inputs.

The physical streams are previously recorded sensor inputs, cast from float64
to float32. This is **open-loop replay**: the new logits do not alter those paths.
There are no new arrivals or reward outcomes to report.

The [NumPy verifier](../experiments/embodiment/verify_food_core.py) checks every
fast state, slow state, pooled feature, logit and probability. The same exports
and fixtures also passed in the isolated embodiment environment with Python
3.12.14 and NumPy 2.5.3, without importing PyTorch; native preparation used
PyTorch 2.8 and NumPy 2.2.6. The
[isolated record](../reports/food-core/isolated-parity.json) preserves all errors
and package identities.

| Interface | Largest absolute error across all checked values | Greedy actions |
|---|---:|---|
| Initial, seed 42 | 1.19 × 10⁻⁶ | All equal |
| Language-trained, seed 42 | 7.15 × 10⁻⁷ | All equal |
| Initial, seed 43 | 1.43 × 10⁻⁶ | All equal |
| Language-trained, seed 43 | 7.15 × 10⁻⁷ | All equal |

Declared comparison tolerances are `atol=5e-6`, `rtol=5e-5`. Agreement establishes
implementation compatibility on these inputs, not useful action selection.
The source-model lexical probe is small; the full sensory replay is not a
replacement language benchmark.

## Reproduce the numerical replay

Download [the complete interface archive](https://flm.kuber.studio/research/food-core-interface.zip)
and extract it to a new directory. It contains four weight exports, the complete
native reference arrays, the NumPy runtime and verifier, provenance, licenses
and the bound numerical sources. With NumPy installed, run from that directory:

```console
python verify_food_core.py . --output fresh-parity.json
```

Use a new output filename; prior records are preserved. Set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before launching for the declared
single-thread configuration. The verifier loads the compressed reference arrays
repeatedly and can take several minutes. Neither a training corpus nor the body
simulator is required. The [release inventory](../reports/food-core/release.json)
records the archive hash and every member's size and hash.

## What remains to test

One recurrent transition occurs per sensory frame. The source physical records
were sampled at 100 Hz, but language training learned per-token dynamics: this
replay does not recover biological time constants. Any closed-loop experiment
must declare its action clock, action-to-motor mapping and reset policy.

The primary planned comparison keeps the core frozen and fits the same new
interfaces with matched downstream exposure for each initial/trained pair.
Evaluation must retain failures and compare before/after changes on reserved
layouts with counterbalanced rewards, omission and reversal controls. A topology
claim additionally needs matched rewired language cores. None of those fits or
evaluations has run here. This still uses the truncated ranked subset, with no
retained Kenyon cells; it does not implement a mushroom-body learning circuit.
