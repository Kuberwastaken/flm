# Before food adaptation: a declared language-core physical reference

Kuber Mehta · 11 September 2026. This document fixes the following **24 physical
trials before execution**. It is a baseline for a future food-adaptation study,
not that study's training or held-out evaluation protocol. No corpus is opened
and no model or adapter is fitted. The registered BabyLM queue remains running;
the complete-group selection comparison keeps subsequent training priority.

## Fixed inventory

Use all four exports from [the verified sensory interface](FOOD-CORE-INTERFACE.md):
`initial-s42`, `language-s42`, `initial-s43`, `language-s43`. Their source archive
SHA-256 is `43e43f9711c3b9a51aa07cc2fbabd272c3194284af0b0c29312f2bfb88eca142`.
The paired initial cores are reconstructed original WikiText initializations;
the language cores are the released step-6,000 checkpoints. Every new adapter
has seed 711 and zero food-adaptation updates. No alternative adapter seeds,
core checkpoints or action scales are selected after viewing these trials.

Each core runs these six cases in the order below, for **24 trials** total:

| Case | Odor A position | A sugar | B sugar | Odor intervention |
|---|---|---:|---:|---|
| `odor-a-left` | (8, 3, 0) mm | 1 | 0 | None |
| `odor-a-right` | (8, −3, 0) mm | 1 | 0 | None |
| `odor-a-left-reversed` | (8, 3, 0) mm | 0 | 1 | None |
| `odor-a-left-neutral` | (8, 3, 0) mm | 0 | 0 | None |
| `odor-a-left-missing` | (8, 3, 0) mm | 1 | 0 | Zero all four sensed odor channels |
| `odor-a-left-repeat` | (8, 3, 0) mm | 1 | 0 | Exact repetition of the first case |

B occupies the opposite y position. Source fields, virtual contact geometry,
body initialization and sensor attachment match the already completed
[scripted reference](FOOD-APPROACH-REFERENCE.md). That seven-case study, including
its straight-policy control, is preserved unchanged and is not rerun here.
These are specified diagnostic layouts, not independent biological replicates
or previously unseen evaluation layouts for a trained food model.

## Clock and motor mapping

Each trial initializes a new FlyGym 2.1.0 body with 42 actuated joint DOFs and
the existing upstream hybrid gait controller, seed 17. Run 20,000 physics steps
at 0.0001 seconds per step, giving 2 seconds. Sample at time zero and every 100
steps thereafter: 201 observations. Each observation's high-level command drives
the next 100 physics steps; the final observation receives a recurrent update
but its command is not applied beyond the trial horizon.

The six geometric sensor channels are cast to float32 and passed to the core.
Reset both recurrent states to zero at trial start, then run exactly one
transition per observation. Do not reset between commands or on source contact.
The softmax output order is right, straight, left. Deterministic steering is:

```text
turn = 0.4 × (p_left − p_right)
descending drives = [0.8 − turn, 0.8 + turn]
```

No action is sampled. This is the convex mixture of the calibrated right,
straight and left drive pairs. It avoids treating an argmax label as the full
neural output. The chosen scale is fixed, rather than tuned to produce more
visible path differences. Higher second drive turns toward positive y under the
existing calibration. The upstream controller, rather than FLM, supplies leg
coordination, adhesion and mechanical feedback.

A shared engineered stop latches when the `source_contact` sensor reaches 0.5;
both descending drives then remain zero. The core continues updating. Sugar is
an input but neither determines the stop nor updates weights. The model receives
no body position, source label, reward identity, trial phase or target direction.
The source record can include those quantities for independent diagnostics.
The 100 Hz task clock does not identify biological neural time constants.

## Retain and audit

Record all body positions, generalized coordinates, thorax orientation, raw
odor, six sensors, contact masks, logits, probabilities, pooled features, both
full recurrent states and actual descending commands at every observation.
Record first contact with any source, its sugar value, all contacted sources,
minimum distance to each source, final thorax position and posture diagnostics.
A no-contact latency is right-censored at the last recorded observation; never
replace it with a successful arrival time. Preserve partial arrays and explicit
failure details for simulation errors. Do not quietly rerun a failed trial.

Freeze an identity with protocol, code, environment, core bundle and source
hashes before the first body trial. Verify the same identity before each trial.
Completed trials may resume only with matching identities and file hashes;
orphan partial files require investigation. Keep all six conditions for all
four cores in the final summary regardless of their outcomes.

Audit geometric sensors independently from stored body positions and replay
each complete core trace from its recorded inputs. Recompute every command and
outcome. Check each exact repeat on every array. Reversal and neutral pairs are
compared descriptively, not required to remain equal: contact sugar can change
recurrent activity, even though the common stop has already latched. Preserve
all differences and failures. Replay verifies the controller on recorded inputs;
it is not an independent reimplementation or rerun of body physics.

## Interpretation

The comparison is initial versus language-trained recurrence behind identical
random, unadapted interfaces. A changed path does not establish a useful transfer
effect or a biological food response. Any source arrival is contingent on this
mapping, these layouts and one gait seed. There is no after-food-training condition
yet, no anatomical-versus-rewired comparison here, and no language test score.
The actual food-training protocol still needs matched downstream exposure,
checkpoint timing, reserved layouts and a predeclared before/after contrast.
