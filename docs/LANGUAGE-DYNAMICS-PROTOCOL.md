# What changes in FLM's recurrent dynamics after language training?

Declared on 11 September 2026 (Asia/Kolkata), before running this diagnostic.
This uses the two already completed WikiText full-FLM references. It does not
fit models, read corpus tokens, use the pending computation controls, or change
any completed language protocol. Prior language results are already known.

The question is whether, and how, language optimization changed the response
of the recurrent state to a controlled input. This is a model-dynamics
diagnostic, not a language benchmark, memory-capacity test, behavioral transfer
study, anatomical-advantage test, or biological experiment.

## Model states and interventions

Use the same measured 1,024-neuron computational subset, node ordering and
4,096-token language configuration as the completed WikiText comparison.
Bind the selected full-model checkpoint hashes for training seeds 42 and 43
from the frozen language computation identity. Verify complete tensor/config,
graph and pooling inventories before analysis. Corpus caches are neither
loaded nor needed; tokenizer metadata is used to restore the original models.

The recurrent parameters are `edge_log_gain`, `alpha_logit`, `beta_logit`, and
`recurrent_logit`. The original constructor initializes these identically
across seeds. Verify that fact, then use one shared initial core; it is not two
independent replications. For each trained seed create three core states:

| State | Edge magnitudes and recurrent gain | Fast/slow time constants |
|---|---|---|
| Trained | Trained checkpoint | Trained checkpoint |
| Edges/gain only | Trained checkpoint | Initialization |
| Time constants only | Initialization | Trained checkpoint |

Together with the shared initial state, this gives seven conditions. Swaps
are acute parameter interventions, not retrained language models. Group the
edge parameters with recurrent gain explicitly; no separate edge-versus-gain
attribution is made. Nonlinear interactions mean the two swap effects need
not add to the trained effect. Keep every condition regardless of its outcome.

The diagnostic calls the existing `FLM.transition` directly. It bypasses learned
embeddings, input projection/bias, pooling, LayerNorm and lexical readout.
Consequently it isolates these four recurrent parameter groups, while removing
the model's normal text-dependent operating point. It does not measure word
knowledge, next-token accuracy, or a fly responding to language.

## Fixed pulses and observations

Generate eight independent Rademacher directions, one sign per selected neuron,
using NumPy PCG64 seed 91011. Each direction has RMS exactly one. Store the
directions and their hash before measurements; use the same directions for all
conditions. Inject each direction as an external drive for one transition from
zero fast/slow state, at amplitudes 0.001 and 1.0. The first is a small-input
probe of linear behavior; the second tests a finite nonlinear response. Neither
amplitude is calibrated to biological units or ordinary token drives.

After the pulse, use zero external drive for 256 transitions. Time 0 denotes
the state immediately after the pulse; retain observations 0 through 256.
The unit is one model update, not milliseconds. Use float32 nonlinear states
and the source model's float32 constants on one CPU thread. Record per-direction
fast, slow and joint RMS at every observation, with joint RMS defined as the
square root of the mean squared state across both state groups and all neurons.

For each direction retain peak joint RMS and its first time, final joint RMS,
the sum of squared joint RMS divided by squared pulse amplitude, the fraction
of this energy at observations 64..256, and the first post-peak observation
after which joint RMS stays at or below 1% of its peak for the remaining
observed horizon. Report a null value if that last condition is not observed;
do not extrapolate a decay time. These are finite-horizon response descriptors,
not performance scores. Larger or longer responses are not presumed better.

Store actual fast/slow neuron states for the preselected direction 0 at times
0, 1, 2, 4, 8, 16, 32, 64, 128 and 256. This direction is fixed before results,
not selected for visual effect. Retain the original anatomical body IDs so
these states can later be mapped to the subset's 3D positions.

## Linearization and numerical checks

Write the zero-drive update with column state vectors as

    h' = (I - D_alpha) h + D_alpha tanh(g W h)
    s' = (I - D_beta) s + D_beta h'.

At the zero state, the fast Jacobian and joint Jacobian are

    A = I - D_alpha + g D_alpha W
    J = [[A,             0],
         [D_beta A, I - D_beta]].

Thus the joint eigenvalues are the fast eigenvalues together with 1-beta_i.
Use the actual normalized float32 W, alpha, beta and gain, converted to float64
for this calculation. Compute all fast eigenvalues with `torch.linalg.eig`.
Record them, the spectral radii of A and J, and the leading eigenpair's relative
residual `||A v - lambda v|| / ((||A||_F + |lambda|) ||v||)`. Require a finite
residual at most 1e-10; retain failures rather than silently changing solvers.

The spectral-radius criterion describes the linearized system at zero drive.
It is not a global nonlinear stability certificate or an echo-state-property
test under arbitrary inputs. State boundedness also differs from convergence:
the original update from zero keeps each nonlinear state in [-1,1] because it
mixes bounded tanh outputs and previous states with coefficients in (0,1).

For both pulse amplitudes, also evolve the zero-state linearization in float64:
start at `h_0 = alpha * pulse`, `s_0 = beta * h_0`, then apply A and the slow
update. Record joint RMS deviation from the nonlinear trajectory divided by
pulse amplitude at every observation. Do not require a small pulse to remain
linear for the whole horizon: an amplified perturbation can leave that regime.
This records approximation error rather than assuming linearity.

Before measuring real cores, fixture tests must check the analytic Jacobian
against autograd, the block-triangular spectrum against a direct full-J solve,
zero-input invariance, isolated-unit closed-form decay, prefix-free pulse
initialization, exact parameter-group swaps, nonlinear state bounds, and
preservation of model tensors/RNG. Run no optimizers in this diagnostic.

Report alpha/beta distributions and recurrent gain. Bare leak half-lives
`log(0.5)/log(1-alpha)` and `log(0.5)/log(1-beta)` describe uncoupled decay factors,
not the memory duration of the coupled network. Keep that distinction explicit.

## Reproducibility and interpretation

Commit this protocol, numerical code and fixture evidence, then freeze their
hashes with graph/tokenizer/reference identities and pulse bytes. Save every
condition's curves, selected actual states, recurrent parameter tensors,
eigenvalues and descriptive summaries. Verify hashes and replay the recorded
conditions before drawing findings. Keep initial, trained and intervened
conditions distinguishable in tables and plots. There are two fitted language
initializations and eight probe directions, not sixteen independent trainings.
No inferential confidence interval, optimum, or superiority claim is declared.

This diagnostic can identify response changes associated with language training
and locate their dependence on these grouped parameters. A later learned
sensor/action interface and controlled transfer experiment are still needed to
test whether those changes improve behavior. The active six-fit language
computation study and subsequent BabyLM program remain the main training queue.

The analysis uses standard dynamical-systems tools. See Stephen Boyd's
[EE263 eigenvectors lecture, slide 11-34](https://ee263.stanford.edu/archive/eig.pdf)
for discrete linear stability. Herbert Jaeger's
[short-term-memory report](https://www.ai.rug.nl/minds/uploads/STMEchoStatesTechRep.pdf)
and [RNN tutorial, section 7](https://www.ai.rug.nl/minds/uploads/ESNTutorialRev.pdf)
place reservoir memory in an input/output setting. The present pulse diagnostic
does not train delayed-input readouts and must not be labeled memory capacity.
The Jacobian above is derived directly from this repository's `FLM.transition`.
