# Executing paired food-adaptation schedules

Kuber Mehta · 12 September 2026. The [schedule executor](food-schedule/food_schedule.py)
now connects complete model/method/stream inventories to the tested
[durable episode store](food-episode-runner.md). **No official food-adaptation
schedule, physical training budget or result is declared by this implementation.**
BabyLM and the complete-group neuron-selection comparison retain priority.

## What is fixed before execution

`initialize(directory, request, weights)` freezes the full request, every
core's array digest, the exact numerical source files and NumPy version. An
existing identity cannot be replaced by a different request. A request contains:

| Field | Required content |
|---|---|
| Purpose and protocol | Explicit scope and nonempty provenance/declaration binding. These fields do not independently validate an official protocol. |
| Clock and rewards | Physics timestep, action interval, complete episode horizon and terminal reward mapping; no implicit training budget. |
| Models | Named initial/language core pairs with source records. Every pair must contain both roles. |
| Methods | Named learning flags and explicit readout learning rate, eligibility decay, baseline rate and matching action budget. |
| Environments | Complete identities returned by the environment factory, including its declared conditions, seeds and sources. |
| Streams | Ordered training episodes, ordered evaluation probes and checkpoint episode counts, including the initial and final counts. |

Each episode names its phase and environment and supplies its own action seed.
Streams must have the same phase order, number of training/probe opportunities
and checkpoint positions. The executor expands the **entire Cartesian product
of models × methods × streams**; it cannot silently select one model or method
from that request. Within an initial/language pair, sensory and initial action
weights, neuron IDs and pooling must match exactly. Trained recurrent parameters
may differ. Core arrays remain frozen while the declared action head learns.

All models and methods consume the same episode schedule within a stream.
Shared uniform streams do not guarantee identical actions or trajectories:
different policies can map the same draw to different actions. Declared episode
opportunities match, but contacts and failures can change actual steps, decisions
and observation exposure. Those quantities remain in the episode records.

## Training and failure semantics

`train(...)` executes each condition's training episodes in their declared
order. Every episode starts with zero recurrent state and the specified action
and environment seeds. Its initial readout is the accepted readout from the
preceding episode. Checkpoints store that actual readout after the declared
number of episode opportunities, including failed slots.

Durable failures retain their slot and preserve the incoming readout. They are
not retried until successful or silently replaced with additional training.
An interrupted attempt without a terminal result restarts its original inputs,
using the durable store's attempt history. A bounded call can stop between
episodes and later continue the same schedule. It does not choose stopping
points from observed reward.

The training completion record binds every condition, episode terminal hash,
complete attempt-file inventory, outcome, cleanup failure and checkpoint readout. Sources and input arrays are
checked again before a bounded return or completion. Extra episode directories,
altered identities and inconsistent saved controller states are errors.

## Evaluation cannot train or choose a checkpoint

`evaluate(...)` refuses to construct any probe environment until **all training
conditions have completed and their complete records have been replay-checked**.
It also reconstructs and verifies the saved readout chain, rather than trusting
an editable checkpoint summary.

Every probe at every declared checkpoint starts from that checkpoint's readout
and a fresh recurrent state. Learning is always disabled for evaluation, even
when enabled for the corresponding training method. A probe's reward, baseline
or action trace cannot carry into another probe or back into training. The
initial checkpoint provides the new sampled-policy baseline; the previous
continuous-drive physical reference is not substituted for it.

Every model/method/stream/checkpoint/probe combination remains in the final
record, including simulation failures. The executor checks the inventory,
training completion hash, all training/evaluation attempt files, source files and core arrays again before publishing
evaluation completion. Cached terminal episodes are reused without new physics.
These checks support the execution contract; a valid completion record does
not mean that the fly learned successfully.

## Verification and scope

The [unit suite](food-schedule/test_food_schedule.py) uses an explicitly artificial
four-neuron core pair. The second core perturbs the fast time constants; it is **not a
language-trained model**. Two readout rules and two action streams give eight
conditions, with two training episodes per condition and two probes at each
of three checkpoints: **16 training slots and 48 evaluation slots**.

Seventeen tests exercise whole-inventory completion, the evaluation gate, exact
checkpoint chaining, bounded resumption, unchanged evaluation readouts, failed
and interrupted slots, disabled-learning methods, malformed requests, changed
interfaces, missing records, extra late episodes, earlier attempt-file changes
and core changes during the last training or evaluation episode. The [verification record](food-schedule/preparation.json)
retains source hashes and results for both Python environments. They are
software tests with toy transitions, not a food-learning experiment or a new
physical restart measurement.

```powershell
python -m unittest discover -s tests -p test_food_schedule.py -v
.venv-embodied/Scripts/python.exe -m unittest discover -s tests -p test_food_schedule.py -v
```

## What still has to be registered and measured

This module is the execution layer. It does not enforce the scientific quality
of caller-supplied provenance, detect geometric train/test leakage, choose
counterbalanced reward assignments, verify completion of priority language
studies, or measure compute costs. Those remain requirements for the official
registration and physical entry point, before any official adaptation run.

The [food-response plan](food-response-plan.md) still needs the full condition
inventory and adequate core/interface/action/gait seeds, training-only cost
measurements, a justified episode budget, unseen physical evaluation layouts,
reward counterbalancing, neutral/missing-cue probes and a fixed analysis.
Reversal probes must be labeled by the association being tested; returning to
training on the original association is reacquisition, not a passive retention
test. Aggregate outcomes must include wrong contacts, timeouts and failures.

The first frozen-core comparison can test adaptation through this engineered
interface and whether language pretraining changes it. It cannot establish an
anatomical advantage without suitable rewired and other computational controls,
or turn an odor-field task into evidence of biological feeding.
