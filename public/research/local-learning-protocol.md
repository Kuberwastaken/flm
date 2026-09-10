# Forward eligibility and learned choice: first controlled study

Declared before fitting this behavioral model. This is a controlled algorithm
comparison, separate from WikiText/BabyLM language training. It does not establish
that a simulated animal understands language or that a learning rule is exclusive
to a fly connectome.

## Network and local derivative

Use the 256 highest-strength central intrinsic neurons selected by the existing
anatomy-only rule from the verified MaleCNS graph: 6,678 signed edges and 32 pools,
graph SHA-256 `0813d767bf6d7f8b6ef3a22a6df3bf0b1713e63c92a8a90be432bc7d2921306c`.
This is a different compact subset from the 1,024-neuron language experiment.
Retain FLM's fast/slow recurrence, incoming absolute-weight normalization, time
constant bounds and gain bounds. Replace vocabulary embedding with four sensory
inputs and the lexical output with a two-action readout of 32 fast/32 slow pools.
Use LayerNorm on pooled state; it is an engineered readout, not local biology.

For postsynaptic neuron i and presynaptic neuron j, let

    v_ij = base_ij exp(theta_ij), W_ij = v_ij / sum_k |v_ik|
    q_i = sum_j W_ij h_j(previous)
    z_i = tanh(A_i x + b_i + g q_i)
    h_i = (1-alpha_i) h_i(previous) + alpha_i z_i
    s_i = (1-beta_i) s_i(previous) + beta_i h_i.

All six parameter groups in the core may learn: input weights/bias, edge log
magnitudes, alpha/beta logits, and the shared gain logit. Project edge logits to
[-3,3] after updates. Signs and graph topology remain fixed. The differentiable
incoming normalization gives the direct edge derivative

    d q_i / d theta_ij = W_ij h_j(previous) - |W_ij| q_i.

Our forward approximation keeps each neuron's own temporal Jacobian

    J_i = 1-alpha_i + alpha_i (1-z_i^2) g W_ii
    e_h(parameter) <- J_i e_h(previous) + direct_h(parameter)
    e_s(parameter) <- (1-beta_i) e_s(previous) + beta_i e_h(parameter)
                      + direct_beta(parameter).

Input direct terms are alpha_i(1-z_i^2)x; bias replaces x with 1. Alpha's
direct term is (z_i-h_i(previous)) d alpha_i/d logit. Gain's is
alpha_i(1-z_i^2)q_i d g/d logit. Beta's direct slow term is
(h_i-s_i(previous))d beta_i/d logit. All non-edge traces are associated with
their postsynaptic neuron; shared gain contributions are summed over neurons.
At the final decision, multiply traces by the instantaneous readout derivatives
with respect to h and s, then sum over the batch. Do not backpropagate through
stored earlier states. The approximation omits indirect temporal routes through
other neurons; it is not the exact BPTT gradient in a recurrent network. Verify
exactness against autograd when those routes are absent and verify the normalized
edge direct derivative independently. Keeping a forward trace does not remove
its memory cost: this version stores two traces per parameter and batch example.

## Learning conditions

Same initialized network, data stream, batch 8 and plain SGD learning rate .03,
no momentum/weight decay, global gradient clipping at 1, for seeds 17, 29 and 41:

- `bptt`: exact full-episode supervised backpropagation through time.
- `reservoir`: supervised output/LayerNorm learning; fixed recurrent core/input.
- `eligibility`: supervised readout error with the forward local core traces.
- `instantaneous`: the same supervised rule, clearing eligibility history at
  every input frame. This is the trace-removal control.
- `reward`: sample a categorical action; reward 1 for the correct direction,
  otherwise 0. Use -(reward-baseline) log probability with a scalar exponential
  reward baseline (.95 previous + .05 batch reward), initially .5. The advantage
  uses the previous baseline. Readout credit is exact for this sampled action;
  core credit uses the same approximate traces. In a binary task, correctness
  feedback also reveals the other action's status; this is not evidence of
  learning from strictly less target information than the supervised condition.

LayerNorm and the readout use ordinary instantaneous derivatives in every
condition. The eligibility/reward conditions still use engineered error routing
through these output weights. They are not claims of fully biological training.
All methods use identical initialization within a seed; action sampling has a
separate RNG so it cannot change the sensory stream. Weights update after each
batch of complete episodes, not within an episode.

## Task, exposure and evaluation

Each episode presents one of two balanced cues for two frames, followed by a
uniformly sampled integer delay from 4 through 12 frames, then a query frame.
Four inputs are left cue, right cue, independent Gaussian distractor (SD .2 on
every frame), and query. A batch shares its delay and has four examples of each
cue in randomized order. Synthetic sensory values are task stimuli, not language
training data. Reset h, s and eligibility at each episode; retain learned weights.
There is no biological millisecond interpretation of an input frame.

Train 900 updates (7,200 episodes) in fixed phases: original mapping for updates
1-300, reversed mapping for 301-600, original mapping again for 601-900. The network
receives no explicit phase flag. Each method gets the same stimuli, targets,
schedule and exposure within a seed. Save states at 0, 300, 600 and 900. Do not
select checkpoints or tune settings from the following probe scores.

Every 100 updates and initially, evaluate fixed balanced panels of 256 episodes
at delays 4, 8, 12, 24 and 48. Use a separate panel RNG seeded 7001+delay, with
identical stimuli for every model/checkpoint. Report greedy action accuracy and
cross-entropy for both mappings from the same probabilities; delays 24 and 48
are extrapolation. These repeated probes are diagnostics, not untouched final
test data. Record each prediction and the actual fast/slow state for original
zero-distractor cue episodes at delay 8, along with parameter-group changes.

The initial study reports all three seeds, curves through reversal, long-delay
performance, update/trace accounting and failures. It does not compare the
anatomical graph against a rewired graph, equalize wall time, or establish an
anatomical advantage. Those are separate controls in the larger program.

## Physical action assay

Use seed 17 for every method, the predeclared delay-8 zero-distractor episodes,
and checkpoints 0/300/600/900. Decode action by argmax. Action 0 drives the
calibrated hybrid controller with [0.4,1.2], action 1 with [1.2,0.4]. Replay one
simulated second from the same initial body state, physics seed 17, for both cues.
Record neural probabilities/state, actual chosen command and physical trajectory.
This attaches a learned high-level choice to a designed gait controller. The
physical simulator supplies the resulting body motion; the neural model has no
online proprioception or learned balance in this first assay. It is separate from
the browser's illustrative pose and from language-to-motor transfer.

## Research basis

Bellec et al.'s [e-prop paper](https://doi.org/10.1038/s41467-020-17236-y)
motivates forward eligibility multiplied by a neuron-specific learning signal.
Its [earlier mathematical account](https://arxiv.org/html/1901.09049v2) explicitly
also treats artificial recurrent neurons. Our equations above are a derivation
for FLM's rate dynamics, normalization and slow state, not a reproduction of the
paper's spiking experiments or accuracy claims. A local temporal derivative is
combined with an approximate learning signal; omitting network-wide temporal
credit can change the gradient and performance. The physical assay uses the
documented [NeuroMechFly/FlyGym hybrid controller](https://neuromechfly.org/).
