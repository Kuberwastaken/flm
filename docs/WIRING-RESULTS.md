# Wiring, context and local learning: completed diagnostic study

The 60-run study is complete. Measured wiring does not have a consistent
advantage across the two tasks and five learning rules. Its strongest long-delay
cue results do not carry over to a uniform context-task advantage. These are
artificial memory diagnostics, not a language benchmark.

Every condition uses 256 retained neurons, 6,678 edges and 8,729 allocated
parameters. Five learning rules are crossed with measured/rewired topology and
two tasks, using three paired initialization/null seeds. All runs receive 900
updates and 7,200 episodes. The cue-to-answer rule reverses at update 301 and
returns at 601. The context task requires combining a remembered cue with a
later context bit, with delays on both sides of the context. This changes both
the required computation and the number of frames per episode.

## Results without selecting favorable seeds

Final update 900, accuracy percentages averaged over all three declared seeds:

| Learning rule | Cue, delay 48: measured | Cue, delay 48: rewired | Context, delay 8: measured | Context, delay 8: rewired |
|---|---:|---:|---:|---:|
| Full BPTT | 100.00 | 66.67 | 63.02 | 76.04 |
| Fixed recurrent core | 86.46 | 55.73 | 63.15 | 44.53 |
| Supervised eligibility | 83.33 | 50.00 | 53.52 | 57.03 |
| No trace history | 50.00 | 66.93 | 57.68 | 55.34 |
| Reward plus eligibility | 50.00 | 50.00 | 66.67 | 50.00 |

These are two declared panels, not a selection rule or the entire result. The
website exposes all five delays (4, 8, 12, 24, 48), ten checkpoints (0 through
900), two tasks, five rules and three seed pairs: 3,000 seed-level accuracy
observations. Final delay-48 context accuracy is near chance in most conditions;
the measured fixed-core mean is 57.55%, and neither BPTT nor supervised
eligibility demonstrates useful long-delay context performance in that panel.

Do not interpret a three-seed mean as a robust ordering when ranges are wide.
The context BPTT result in particular defeats a blanket claim that measured
wiring is better for this model machinery. Supervised eligibility also fails
to improve uniformly over the no-history approximation or fixed core. Local
learning remains a candidate to investigate, not an established replacement for
backpropagation.

## What the controls actually hold fixed

Three independent signed double-edge-swap graphs preserve directed in/out
degrees, incoming signed weights, source-sign constraints, original self edges,
node identities, coordinates and pooling. Every chain accepts ten swaps per
edge. They retain roughly 23% of measured edges; reciprocal off-diagonal edges
fall from 67.96% to about 14.74–15.31%. Transferred weights on new edges are
artificial. This null does not preserve every motif, outgoing weighted strength,
spatial distance or a proven stationary graph distribution.

Initialization and sampled sensory streams match across conditions. The 15
measured cue runs reproduce every trained parameter tensor and training-stream
hash from the previous local-learning study. They are a reproducibility bridge,
not 15 additional independent observations.

## Credit-assignment diagnostics

At updates 0, 300, 600 and 900, a fixed supervised probe compares the exact BPTT
core gradient with forward eligibility and the no-history approximation. All
network parameters and random states remain unchanged by measurement. The
archive retains per-component norms, cosine, relative error and sign agreement,
including undefined metrics when the required reference norm is zero.

The plotted context diagnostic uses BPTT-trained checkpoints and delay 8. The
approximate gradients are not exact BPTT, their alignment changes during learning,
and the graph changes that alignment. Alignment is descriptive and is not itself
evidence that a training rule will outperform another. In reward-trained models,
this remains a supervised cross-entropy probe; it must not be called the exact
gradient of the reward estimator.

## Evidence and limits

The [declared protocol](WIRING-LEARNING-PROTOCOL.md) fixes all tasks, seeds,
budgets, panels and diagnostics. The machine-readable result is
`reports/wiring-learning/summary.json`; the public JSON, CSVs, signed adjacency
plots and archive are in `public/research`. The archive contains all 60 full
reports, including every two-class probe prediction, graph files/cards and the
protocol. Figures show seed minima/maxima, not confidence intervals.

The three graph seeds are paired with, rather than crossed across, model seeds
in this small study. Their uncertainties cannot be separately estimated. Panels
are reused diagnostics and have no untouched test status. The body replay study
uses designed gait control and remains separate.

The language question is now addressed by the separate
[language topology protocol](LANGUAGE-TOPOLOGY-PROTOCOL.md): three 1,024-neuron
null graphs fully crossed with two training seeds, plus two retrained slow-state
controls under the original WikiText budget. These toy results do not substitute
for that comparison, whose test results remain pending.
