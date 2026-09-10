# Language training changed the recurrent response

The two completed WikiText FLM models respond differently from their shared
initial core when given the same synthetic pulse. Language training moved the
zero-drive **fast-state spectral radius from 1.0207 to 0.9912 and 0.8722**.
This is a change in the measured model dynamics. It does not demonstrate better
language performance, useful memory capacity, anatomical advantage, or transfer
to the fly's behavior.

![All seven cores responding to two fixed pulse amplitudes](../public/research/figures/language-dynamics-pulses.svg)

The [protocol](LANGUAGE-DYNAMICS-PROTOCOL.md) and numerical sources were committed
before the diagnostic. The [identity](../reports/language-dynamics/identity.json)
binds both completed language checkpoints, the measured 1,024-neuron subset,
and eight fixed pulse directions. Every core received a one-update drive at
amplitudes 0.001 and 1, then 256 zero-input updates. The normal lexical interface
was bypassed. No corpus tokens were read and no new models were fitted.

## What changed, and which parameters account for it?

| Core state | Fast spectral radius | Joint fast/slow radius | Recurrent gain |
|---|---:|---:|---:|
| Shared initialization | 1.020729 | 1.020729 | 1.525000 |
| Trained, seed 42 | 0.991243 | 0.991243 | 1.683331 |
| Edges/gain only, seed 42 | 0.992904 | 0.992904 | 1.683331 |
| Time constants only, seed 42 | 1.021354 | 1.021354 | 1.525000 |
| Trained, seed 43 | 0.872189 | 0.990334 | 1.664763 |
| Edges/gain only, seed 43 | 0.873639 | 0.990566 | 1.664763 |
| Time constants only, seed 43 | 1.022277 | 1.022277 | 1.525000 |

Copying the learned edges and recurrent gain together into the initial core
largely reproduces the fast spectral shift. Copying only the fast/slow time
constants leaves that radius above one. The scalar recurrent gain actually
increased during training. The grouped intervention does not separate the
effect of edge redistribution from gain, and its effects need not add to the
time-constant intervention's effects.

For the small pulse, all eight seed-43 responses met the declared 1%-of-peak
settling rule within 54–57 updates. Five of eight seed-42 responses met it
within 68–214 updates; three had not met it by 256. No initial-core response
met it within that horizon. These are fixed-direction descriptors, not
independent training replications or a statistical superiority result.

The joint radius for seed 43 remains about 0.9903 because the separate slow
state retains a long decay factor. Bare slow-leak median half-lives changed
only from 24.10 updates initially to 23.78 and 23.89; bare fast-leak medians
changed from 1.00 to 0.72 and 0.70. Bare leak factors and coupled-network
response times are different measurements.

## Interpretation and limits

These eigenvalues describe the Jacobian at **zero drive and zero state**.
The [linear spectral-radius criterion](https://ee263.stanford.edu/archive/eig.pdf)
does not certify global nonlinear stability under arbitrary text inputs.
Bounded state is also different from decay to zero. The pulse curves show
responses in the declared finite horizon, including the nonlinear amplitude.

Long-lived activity is not automatically usable memory. This study fits no
delayed-input readout of the kind discussed in
[Jaeger's reservoir tutorial, section 7](https://www.ai.rug.nl/minds/uploads/ESNTutorialRev.pdf).
The two trained checkpoints have similar published language losses despite
their different zero-drive responses. These diagnostics therefore should not
be treated as a ranking of language quality.

The swaps are acute interventions, not retrained controls. The separate
[six-fit language computation study](LANGUAGE-CORE-PROTOCOL.md) remains the
experiment testing whether trainable dynamics and recurrence help prediction.
The completed topology comparison still demonstrates no anatomical advantage
for its selected subset and budget. A learned sensor/action interface and
matched transfer experiment are still needed to test behavioral consequences.

## Reproduction

All seven cases reproduce their metrics and every saved array exactly in the
recorded one-thread PyTorch 2.8.0 CPU / NumPy 2.2.6 environment. The
[replay record](../reports/language-dynamics/replay.json) binds the summary and
all case files; it is evidence of same-environment replay, not independent
retraining. Eleven fixture tests precede the measurements. Source code is in
[`language_dynamics.py`](../flm/language_dynamics.py) and the
[`study harness`](../flm/language_dynamics_study.py).

The [curve CSV](../public/research/language-dynamics-curves.csv) and
[summary CSV](../public/research/language-dynamics-summary.csv) retain all seven
conditions. Case arrays preserve per-direction curves, eigenvalues, recurrent
parameters, anatomical body IDs, and actual neuron states for the preselected
direction at ten observation times. No displayed state was scripted for effect.
