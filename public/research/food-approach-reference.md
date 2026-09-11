# Scripted odor approach: physical interface reference

Kuber Mehta · 11 September 2026. This declaration precedes the seven physical
trials below. It is an engineering reference for the food-response extension,
not an FLM learning, language-transfer or biological feeding experiment.
Existing training, selection, sensory and physical studies remain unchanged.

## What the controller can see

The [food sensors](food-sensor-interface.md) sample actual antenna and foot
body-frame origins. Only their six normalized channels enter the scripted
controller. Source coordinates, body pose, yaw, sugar identity at a distance,
target-bearing errors and evaluation labels are unavailable to it. The upstream
hybrid gait controller still uses its existing mechanical feedback.

The new reference prefers odor A by design; it does not learn this association.
At 100 Hz, its bilateral contrast is `(A_left-A_right)/(A_left+A_right+1e-12)`.
Multiply by 24, clip to ±0.4, then send drives `[0.8-turn, 0.8+turn]` to the
previously calibrated gait interface. Odor B and sugar contact do not steer it.
Any-source contact at or above 0.5 latches both drives to zero. This is a stop
command, not a guarantee that physical motion immediately ceases or feeding starts.
Zero/equal odor produces straight walking without an exploratory search policy.

## Fixed physical cases

Each independently initialized simulation uses gait seed 17, a 0.0001-second
physics step and 2 seconds of motion (20,000 steps). There is no renderer. Source
A is at (8, 3, 0) mm and B at (8, −3, 0) mm, except for the mirrored condition.
Both have Gaussian spreads (5, 4, 2) mm; A emits only channel A with amplitude 1,
and B only channel B. Virtual contact cylinders have radius 0.75 mm and height
0.25 mm above source z. These are declared task volumes, not fruit colliders.

| Case | Odor A location | A/B sugar intensity | Policy / intervention |
|---|---|---|---|
| odor-a-left | Left | 1 / 0 | Odor contrast |
| odor-a-right | Right, source locations mirrored | 1 / 0 | Odor contrast |
| odor-a-left-reversed | Left | 0 / 1 | Same policy; reward identities reversed |
| odor-a-left-neutral | Left | 0 / 0 | Same policy; no sugar source |
| odor-a-left-missing | Left | 1 / 0 | Both sensed odors removed |
| straight-a-left | Left | 1 / 0 | Equal drives, same contact stop |
| odor-a-left-repeat | Left | 1 / 0 | Independent exact-seed repeat of first case |

Parameters and all cases are fixed before outcomes. They are not a random sample
of layouts, held-out learned-task evaluations or independent biological trials.
If this reference fails to approach a source, preserve that result before any
separately declared controller revision; do not tune against these cases and
call them unseen evaluation.

## Measurements and interpretation

Record body origins, generalized positions, thorax orientation, six inputs,
raw odor, geometric contact masks and the actual descending commands at 100 Hz,
including reset and the final endpoint: 201 observations on completion.
Commands at time t precede the following 100 physics steps; the final endpoint
receives no additional physics. Contact is assessed only on this sampling grid;
unobserved contacts between samples are not counted. No privileged diagnostic
is fed back to the odor policy.

Report the first sampled source contact, its latency and sugar intensity, all
subsequently contacted sources, minimum thorax-to-source planar distances,
final position, physical stability and any simulation failure. Mark no-contact
latencies as censored at 2 seconds for complete trials, or at the last valid
sample for failed trials. Keep simultaneous first contacts explicit. A contact
with a neutral source is an observed contact, not sugar success.

Replay all sensor values from recorded geometry and all commands from the
recorded six-channel histories. Compare repeated body states exactly; compare
the reward-reversed and neutral paths to the first case, since changing sugar
cannot affect this policy. Missing odor should match the straight reference's
physical path under the same initialization. Report disagreement rather than
silently substituting a trajectory.

At best these trials demonstrate usable sensor-to-action feedback through an
engineered odor field and gait controller. A reversed-reward failure is expected
from its hardwired odor preference, not evidence that a learned model forgot.
The later paired random-core versus language-trained-core experiment still
needs its own frozen adaptation protocol, unseen conditions and fitted models.
