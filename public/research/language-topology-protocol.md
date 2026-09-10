# Does measured wiring help language prediction?

Declared on 10 September 2026 before fitting these language controls. This is
an exploratory extension of the registered WikiText-2 comparison. Its main
test results and acute mechanism interventions have already been inspected;
this is not a new untouched-test discovery study. The larger BabyLM queue is
paused at saved checkpoints to prioritize this missing comparison.

## Question and comparison

Does FLM's measured 1,024-neuron subset predict held-out language better than
independently rewired graphs with the same recurrent model and matched local
constraints? Separately, does the slow state improve prediction after allowing
the model to retrain without it?

| Condition | Graph seeds | Training seeds | Runs |
|---|---|---|---:|
| Measured wiring, fast and slow state | Original graph | 42, 43 | 2 existing references |
| Rewired, fast and slow state | 101, 103, 107 | 42, 43 for **each** graph | 6 new |
| Measured wiring, slow state disabled throughout training | Original graph | 42, 43 | 2 new |

The measured references are the previously completed 6,000-update FLM runs,
bound to their frozen checkpoint hashes in `reports/wikitext2/selection.json`.
Their learned weights are never used to initialize controls. New models start
from scratch. For each training seed, every condition starts with exactly the
same named parameter tensors and sees the same sampled text. Graph buffers
differ only where the protocol requires. Verify the current trainer against
the reference source on a short deterministic optimizer replay before fitting.
Reuse of the two references does not create additional independent runs.

## Null graphs

Use `data/graphs/central-1024/graph.npz`, SHA-256
`a2f35369a5ed4981b8ef75c92df285f7df576a020582f119a1372bfbedbaebee`:
1,024 retained MaleCNS neuron identities and 76,130 directed weighted edges.
This is a selected subset, not a whole fly brain.

Use the signed double-edge-swap algorithm in `flm/wiring_controls.py`.
In postsynaptic-row/presynaptic-column notation, propose
`(i,j), (k,l) -> (i,l), (k,j)` only when sources `j,l` have the same sign.
Reject duplicate edges, new self edges, unchanged proposals and invalid
endpoints. Freeze every original self edge at its original weight. Stop at
exactly ten accepted swaps per original edge, failing if this needs more than
100 proposals per edge. Independently initialize all three chains from the
measured graph with NumPy generator seeds 101, 103 and 107; do not select graphs
based on model performance.

Validate exact per-node directed in/out degrees, each neuron's source sign,
each target's signed incoming weight/contact multiset, node order, positions,
pool membership and original self edges. Weights remain attached to their
original postsynaptic slots. New contacts are artificial transferred magnitudes,
not measurements of those newly connected neuron pairs. Outgoing weighted
strength, distance, reciprocity, connected components and higher motifs need
not be preserved. Report edge overlap, reciprocity, strong components and
acceptance counts for every null. Ten swaps per edge is a finite chain budget;
uniform sampling and mixing are not established.

The contrast concerns higher wiring organization conditional on these preserved
features. It does not test whether all anatomical information is useful versus
none, because node identity, signs, degrees, input magnitudes and pooling remain
anatomically derived in the controls.

## Matched learning and slow-state control

Keep `docs/WIKITEXT-PROTOCOL.md` unchanged: official raw article partitions;
the same training-only 4,096-ID byte-BPE tokenizer; 600 training articles; batch
16, sequence 96, first 16 positions excluded from loss; 6,000 AdamW updates;
9,216,000 presented tokens; warmup 100, learning rate 0.002 to 0.0002 by cosine
decay; weight decay 0.01; gradient clipping 1; four CPU threads; seeds 42/43.
Use the exact original train/validation cache hashes and fixed validation
prefixes. No extra tuning, early stopping, changed tokenizer or test feedback.

All models allocate 600,003 parameter entries. The `no_slow` variant sets slow
state to zero from the first update and retrains all remaining effective
machinery. Its 1,024 beta entries are disconnected from the objective. Keep
the original pooling/readout and LayerNorm widths to avoid an additional
architecture change. Normalization can make the nominal slow feature columns
nonzero after centering, so those readout columns are not necessarily unused.
Report this allocated-versus-effective distinction. This tests the implemented
slow-state branch, not all possible capacity-matched memory alternatives.

Save optimizer state and sampler/PyTorch RNG state at each validation checkpoint
and resume only matching identities. Record protocol, numerical source, graph,
tokenizer and cache hashes. Verify reference source compatibility, shared
initial parameter hashes, sampled-token stream hashes and final exposure/RNG.

## Selection, evaluation and reporting

Choose each run's checkpoint by lowest validation bits per UTF-8 byte among
updates 500, 1,000, ..., 6,000, retaining the earliest on an exact tie. All eight
new runs must finish and all ten selected checkpoint identities must be frozen
before scoring any new control on the test split. Existing measured test scores
are already public; reusing them requires exact checkpoint and data identity.

Score all 60 official test articles, resetting state per article and carrying
it within each article, chunk size 96. Exclude inserted boundary IDs and use
exact decoded UTF-8 byte denominators. Report bits/byte, shared-tokenizer
perplexity and per-article losses for every run. Word-token published WikiText
perplexities are not directly comparable.

Primary topology effects are `measured BPB - rewired BPB`, individually for all
six graph/training-seed combinations; negative favors measured wiring. Report
each graph's mean across its two training seeds, each training seed's mean
across the three graphs, and the grand mean. Show all values and ranges; six
contrasts sharing two measured references are not six independent replications.
No binary significance claim or topology-superiority headline is licensed by
three null graphs and two initializations alone.

Primary slow-state effects are `measured BPB - retrained no_slow BPB`, separately
for seeds 42/43 and as their mean. Do not substitute acute inference-time
disabling for this retrained comparison. The design does not estimate a
topology-by-slow-state interaction: null graphs without slow state would need
a separately declared follow-up.

For each individual paired contrast, retain the existing 10,000-replicate
paired article bootstrap (seed 31415). Its 95% interval is conditional on the
two trained checkpoints; it does not represent graph, training or dataset
uncertainty. Do not pool articles across repeated seeds as extra independent
documents. Validation curves at identical updates are secondary descriptive
learning-efficiency evidence. Partial validation snapshots must say that test
comparisons are pending and never select a preferred null graph.

A positive, null or negative result is publishable. A null result means no
advantage demonstrated under this specific subset, parameterization, data and
budget; it is not proof that connectomes never help. Cue/context toy tasks,
body animation and physical command execution cannot answer this language
question. The transformer and GRU comparisons remain visible as separate
architecture benchmarks.

## Scientific context

- Maslov & Sneppen (2002), [Specificity and stability in topology of protein
  networks](https://doi.org/10.1126/science.1065103): motivates degree-preserving
  edge-swap nulls; our sign, weight-slot and self-edge constraints are explicit
  additions for this model.
- Lappalainen et al. (2024), [Connectome-constrained networks predict neural
  activity across the fly visual system](https://doi.org/10.1038/s41586-024-07939-3):
  evidence in a visual-system activity-prediction setting, not language.
- Dhiman (2026), [Topological Sensitivity in Connectome-Constrained Neural
  Networks](https://arxiv.org/abs/2604.04033v1), preprint: a short-horizon flyvis
  study reports that shared from-scratch initialization and degree-preserving
  controls weaken apparent topology advantages. Its task and training horizon
  differ from FLM; this is motivation for controls, not a predicted FLM result.
