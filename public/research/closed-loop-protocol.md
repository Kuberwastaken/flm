# Online pose feedback through the learned choice network

Author: Kuber Mehta. Declared before running the physical trials below. This is
an engineering and transfer assay using already-inspected cue-learning models.
It is separate from the ongoing language topology experiment, which retains
priority for training resources. No model is fitted or selected in this assay.

## Question and scope

Can the existing learned cue-to-action mapping steer the physical model when
its current position and heading determine the next cue? Does removing that
online pose information change tracking? Compare trained recurrent dynamics
with the shared untrained initialization and with a trained fixed-core readout.

This uses the 256-neuron, four-input, two-action sensory-choice FLM. It does not
use the WikiText or BabyLM language weights. The sensor is an engineered encoding
of simulator ground truth, not reconstructed fly vision, proprioceptive neurons
or biological navigation. The upstream hybrid controller still supplies gait,
adhesion and leg/contact feedback. A correct turn does not demonstrate learned
joint control, language comprehension or an anatomical advantage.

## Fixed models and runtime

Use training seed 17 and the original measured 256-neuron graph. Compare the
shared initial checkpoint (BPTT update 0) and final update 900 for BPTT, fixed
core/readout training (`reservoir`) and supervised forward eligibility. These
methods are chosen from the previous published comparison; this is not an
untouched evaluation. Check that all three initial model tensors agree before
using a single shared initialization. Do not add or remove conditions based on
physical outcomes. Reward/instantaneous learners remain in the earlier study.

Export the effective normalized recurrent matrix, time constants, input and
readout tensors to a NumPy inference package. Record original checkpoint and
graph hashes. Before physics, verify fast state, slow state, logits and argmax
against the original PyTorch model on a fixed random sensory stream, resets and
the two original zero-distractor cue episodes. Apply the same tolerances in the
isolated FlyGym environment: absolute 2e-6 and relative 2e-5. This is numerical
parity within a declared tolerance, not bitwise tensor computation equivalence.

## Sensory and action interface

The body starts at the calibrated pose with physics seed 17, on the same flat
ground. A waypoint is a virtual target coordinate, not a physical object or
rendered visual input. There are three two-second scenarios:

- `positive`: target at world position (40, 20) mm throughout.
- `negative`: target at (40, -20) mm throughout.
- `switch`: (40, 20) mm until 1.0 s, then (40, -20) mm.

At the start of each control cycle, compute the wrapped bearing error
`atan2(target_y - y, target_x - x) - yaw`, in [-pi, pi). Nonnegative error encodes
cue 0; negative error encodes cue 1. Current position and heading come from the
thorax pose before advancing physics. A zero-distance target is invalid; the
far waypoint locations avoid intentionally testing arrival/stop behavior.

Each cycle contains eleven neural frames, one every 5 ms of simulator time.
Present the cue for two frames, eight blank delay frames, then one query frame;
the distractor channel is always zero. Reset fast/slow state at the beginning
of each cycle, matching the original episode training. The neural frame duration
is an engineered control clock, not a biological membrane timescale. Hold the
previous descending signal until the query produces a new decision. The first
cycle starts with [1, 1]. Use a 50 ms sensor-to-query delay and a 55 ms cycle.

After the query, argmax action 0 selects [0.4, 1.2]; action 1 selects [1.2, 0.4].
When the sampled absolute bearing error is at most 0.10 rad, use [1, 1] instead.
This straight-ahead deadband is an engineered wrapper shared by all conditions;
the network has only two learned actions. Do not use a newer pose in this gate.
The physical gait controller executes at the unchanged 0.1 ms MuJoCo timestep.

## Controls and complete case set

Cross each of four neural checkpoints with three scenarios and two pose modes:

- `live`: sample actual thorax position and heading at each cycle start.
- `frozen`: use the initial thorax position and heading at every cycle start,
  while preserving the same time-dependent target schedule. Low-level gait
  feedback remains active. This removes pose feedback only from the high-level
  neural interface, without hiding the externally specified target switch.

Add the scripted sign-of-error reference with live pose in all three scenarios.
It receives the same sampled error, decision delay, initial straight command,
deadband and motor interface. It bypasses neural inference. These are 27 distinct
physical trials. Also repeat `eligibility/live/switch` once from scratch with the
same identity to check exact equality of recorded physical trajectories and
neural decisions. The repeat is a reproducibility check, not another seed.

The single physical seed and single training seed do not support population
confidence intervals. Several trained methods may choose identical actions and
therefore follow identical paths; retain and identify that outcome. All cases
run to two seconds unless physics becomes nonfinite or raises an error. Such
failures remain explicit; do not silently drop trials or change the duration.

## Measurements and reporting

Record the initial pose and then pose, quaternion, generalized positions,
contacts, desired target, true angular error and applied descending command
every 1 ms. At every neural frame, record the four actual sensory inputs,
fast/slow state and output logits; distinguish query decisions from intermediate
readouts. At each cycle record sampled pose/error, target, chosen action,
probabilities, decision time, motor command and whether the deadband applied.

Primary diagnostic: time-mean absolute true bearing error across the complete
two-second trajectory, using trapezoidal integration of the 1 ms records divided
by two seconds. Also report final absolute error, final target distance,
change in distance to the current target within each scenario phase, heading,
minimum thorax height and up-axis component, contact count and command switches.
Treat orientation/height as diagnostics, not a validated fall detector. Report
all 27 conditions and the matched live-minus-frozen error differences. A
negative difference favors online pose feedback, conditional on this wrapper.

Retain full NPZ trajectories locally and publish compact records/CSV traces,
controller package hashes and actual neural state. Record one video chosen in
advance: `eligibility/live/switch`, first execution, quarter-speed playback.
Do not choose the nicest trajectory after seeing results. Use one physical
worker with numerical thread pools limited to one while topology jobs run;
contended wall times are operational logs, not comparative speed benchmarks.

## Interpretation and next step

This closes the high-level sensor/action loop missing from the earlier offline
choice replay. It can show whether the already-learned cue mapping transfers
through this interface, and whether online pose information improves tracking.
It cannot show that the recurrent core is necessary unless it beats the
fixed-core/readout control, or that anatomy is useful without matched wiring
controls. The scripted reference exposes how much behavior the wrapper supplies.

A subsequent language-to-control study must introduce an explicit text/sensory
interface and compare the same readout adaptation using untrained and
language-trained cores. That study remains open; this result must not be used
as a substitute. Larger language training resumes after the current topology
comparison is complete and interpreted.

The physical interface follows the official
[FlyGym 2.1 turning tutorial](https://neuromechfly.org/tutorials/4d_turning_controller/)
and the existing verified calibration. Model learning and its mathematical
approximation are specified in [the original learning protocol](LOCAL-LEARNING-PROTOCOL.md).
