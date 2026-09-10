# Wiring, delayed context and forward credit

Declared 10 September 2026 before fitting this extension. The preceding simple
cue study has already been inspected. This extension is exploratory research,
not an independent confirmation of a hypothesis chosen before that study.

## Questions

1. Does the measured 256-neuron topology outperform a degree/sign-matched
   randomized control under the same learning rule and sensory exposure?
2. Does adding a delayed context cue expose differences that the simple cue
   task, often solved by a fixed readout, did not?
3. How closely does the implemented local temporal approximation align with
   the exact supervised gradient at the same network state?

These questions concern artificial rate networks and numeric sensory tasks.
They do not test language quality, biological validity, learned gait, or whether
a real fly understands language. The ongoing BabyLM study remains separate.

## Graph control

Use `data/graphs/central-256/graph.npz`, SHA-256
`0813d767bf6d7f8b6ef3a22a6df3bf0b1713e63c92a8a90be432bc7d2921306c`:
256 nodes, 6,678 directed edges, 32 fixed readout pools. The null graphs keep
the same neuron identities, coordinates, signs, pools and parameter allocation.
Their edges are artificial and must never be represented as measured synapses.

For two non-self edges `(i <- j)` and `(k <- l)`, swap presynaptic endpoints only
when `i != k`, `j != l`, `sign(j) == sign(l)`, neither proposed edge is a self
edge, and neither already exists. Freeze all original self edges. Leave row
indices, weights and contact-derived magnitudes in their original array slots.
Thus each node's in-degree, out-degree, positive/negative incoming counts,
incoming signed weight multiset and exact self weights remain unchanged.
Outgoing weighted strength, spatial distance, cell-type coupling and higher
motifs are not constrained. The two-step eligibility Jacobian's explicit
self-edge term is not directly removed by the control.

Request ten accepted swaps per retained edge, with at most one hundred
proposals per edge. Fail if that target cannot be reached. This finite swap
chain is a reproducible null construction, not a proof of uniform sampling or
mixing. Record proposals, acceptance, exact preserved invariants, overlap with
the original edge set, reciprocal edge fraction and strongly connected
component sizes. Use graph seeds 101, 103, 107 paired respectively with model
seeds 17, 29, 41. All methods and tasks share the corresponding graph. Three
paired initializations/null realizations do not separately estimate uncertainty
over initialization and null topology.

The original `flm.graph.rewire` used in earlier language tooling is unchanged.
This stricter, self-edge-preserving control is implemented in a separate module
and identified explicitly; it does not silently replace an earlier protocol.

## Tasks and exposure

Both tasks use four numeric channels, eight balanced episodes per batch, a
Gaussian distractor of standard deviation .2 in channel 2 at every frame,
and one final query frame marked in channel 3. Network state and eligibility
traces reset for every episode. No phase indicator is supplied.

**Cue reversal:** reproduce the existing `episode_batch` exactly: one of two
one-hot cue channels is active for two frames, followed by an integer delay
drawn uniformly from 4 through 12, then the query. The target is the cue ID.
This repeats the earlier task as a bridge; it is not a new independent task.

**Delayed context:** sample balanced cue/context pairs `(0,0), (0,1), (1,0),
`(1,1)`, two of each per batch, and shuffle them. The cue appears for two frames
as -1/+1 in channel 0. After a delay D, context appears for two frames as -1/+1
in channel 1. After a second delay D, the query appears. D is sampled uniformly
from 4 through 12 for the batch, giving `2*D+5` frames. The target is cue XOR
context: action 0 for matching values, action 1 for different values. Neither
cue alone determines the answer. This is an engineered conditional-memory
task, not a reproduction of a published animal experiment.

For both tasks, updates 1–300 use the original target, 301–600 invert it, and
601–900 restore it. Each run receives 7,200 episodes. The delayed-context task
has more sensory frames; compare methods and graphs within a task, not by
calling equal episode counts equal recurrent compute across tasks.

Run the full factorial of two tasks, two topologies, five methods and three
model seeds: sixty runs. Rerun the fifteen measured-graph cue conditions under
this harness and verify their final training-stream hash and model tensors
against the preceding study. They are repeated measurements, not extra seeds.

## Learning and initialization

Use `ChoiceFLM` and the five existing learning methods unchanged: BPTT,
fixed-core/readout-only, supervised eligibility, no trace history, and
reward-modulated eligibility. SGD .03, batch eight, global gradient clip one,
no momentum, no weight decay, 900 updates, one PyTorch thread. The network has
8,729 allocated trainable parameters; the fixed-core method updates only 258.
The reward condition keeps its existing separate action RNG and previous
reward baseline. This binary task does not demonstrate less target information.

Within each model seed, every condition starts from identical parameter
tensors. The topology buffers may differ and therefore initial neural states
and predictions need not agree between graphs. Input RNG is reset to the model
seed for every condition; methods/graphs within each task receive identical
stimuli and targets. Record and verify a rolling SHA-256 stream identity.

Do not tune learning rates, initialization, budgets or tasks after inspecting
these results. Any follow-on attempt needs a separate declaration and output
directory. Completing this study does not justify an anatomical advantage on
language; language controls require their own matched training and test scores.

## Diagnostics and retained records

At initialization and every 100 updates, score 256 balanced episodes at delays
4, 8, 12, 24 and 48. Cue panels retain RNG seed `7001 + delay`; context panels
use `17001 + delay`. Record every example's cue/context identity, action
probabilities, stimulus hash, original/reversed cross-entropy and accuracy.
These are repeatedly inspected fixed diagnostic panels, not an untouched test
set and not a checkpoint selection criterion. The final outcome is update 900.

At updates 0, 300, 600 and 900, compare exact BPTT gradients with the existing
supervised eligibility approximation, for both full trace history and the
no-history variant. Use eight balanced episodes, delay eight, RNG seed 27009
for cue and 37009 for context, and the current phase's targets. Evaluate a
float64 copy; do not update it or consume the training RNG. The diagnostic
loss is supervised cross-entropy even for reward-trained networks. This is a
derivative diagnostic, not an alternate reward-training signal.

For the combined recurrent/input parameter group and each component, record
exact/approximate gradient norms, cosine, relative L2 error and fraction of
matching signs where both entries are nonzero. Undefined zero-norm metrics
are null, never silently zero. Readout gradients are excluded from combined
core comparisons because that instantaneous derivative is exact by design.
All gradient measurements precede clipping and parameter updates.

Save recoverable optimizer/RNG/checkpoint state every 100 updates, with atomic
writes and identities binding protocol, graph, implementation and environment.
Reports retain failed cases and all diagnostic predictions. A completed run
must be hash-verified before reuse; no simultaneous writers per run.

Report each seed/null pair and their descriptive mean and range. Paired
measured-minus-null accuracy differences, at every declared delay and phase,
make sign changes visible. Do not hide unfavorable delays, use repeated panels
as independent seeds, present a three-seed range as a confidence interval, or
infer a general anatomical advantage from a single positive comparison.

## Scientific context

- [Maslov and Sneppen, 2002](https://pubmed.ncbi.nlm.nih.gov/11988575/):
  randomized network null models motivate testing structure against constrained
  alternatives. The exact signed/self-edge control above is our construction.
- [Bellec et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/):
  eligibility propagation motivates forward traces and approximate learning
  signals. Our rate-network approximation is derived separately and is not a
  reproduction of their spiking experiments.
- [Lappalainen et al., 2024](https://www.nature.com/articles/s41586-024-07939-3):
  task-optimized connectome models motivate testing the computation supported by
  wiring. Their fly visual-system evidence does not establish an FLM language
  or artificial cue-task advantage.
