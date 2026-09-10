# Language computation controls: follow-up definitions

**Status: prospective, no runs launched.** Finish and report all controls in
the frozen [topology study](language-topology-protocol.md) first. This document
clarifies the next causal questions; it does not amend that protocol or
pre-register a completed experiment.

Learned embeddings, input projection, normalization and readout can predict
text without useful anatomical computation. The separate sensory fixed-core
experiment cannot establish their contribution in language. Three retrained
language comparisons would distinguish different claims:

| Control | Change from language FLM | Question |
|---|---|---|
| Fixed recurrent dynamics | Freeze edge gains, fast/slow time constants and recurrent gain at the shared initialization; train the lexical interface | Does optimizing the recurrent dynamics improve on a fixed dynamical feature map? |
| No lateral recurrence | Remove `W h` at every training and evaluation step; keep the leaky fast state and slow state | Does communication along graph edges improve on independent temporal units? |
| No temporal state | Start both states from zero for each individual token during training and evaluation | Does this model improve on a learned current-token predictor by carrying history? |

The current implementation's `no_recurrence` variant means **no lateral
recurrence**, not no memory: `(1-alpha) h` and the slow update still retain
history. Acute inference-time disabling is a different intervention from
retraining. The ongoing `no_slow` comparison only removes slow-state dynamics
and cannot substitute for any of these controls.

## What “fixed” includes

For the present 4,096-token, 96-dimensional tied-embedding language model,
`edge_log_gain` has 76,130 scalars, `alpha_logit` and `beta_logit` each have
1,024, and `recurrent_logit` has one. Freezing those four groups fixes **78,179**
of 600,003 parameters and leaves **521,824 trainable parameters**. These counts
were checked by constructing the current language model; no fitting was needed.

Embeddings, input projection, LayerNorm and readout remain trainable. Because
the input interface learns, its gradients still pass through the state history.
This is not a readout-only reservoir and does not remove BPTT. It should not be
called parameter matched to full FLM: total stored parameters match while the
trainable degrees of freedom differ. Conversely, a readout-only reservoir would
also freeze the input interface and answer a separate, stricter question.

## Requirements before fitting

Use an independent control harness without editing the numerical sources of
the current study. Declare exact seeds, initialization tensors, token streams,
graph identity, exposure, optimizer groups, validation checkpoint selection
and test gate before training. Report stored, trainable and gradient-connected
parameters separately. For a no-temporal-state implementation, verify that
two different prefixes ending in the same token give identical logits before
claiming it is memoryless. For a fixed core, verify all frozen tensors after
every saved checkpoint; do not infer freezing from an optimizer label.

Start with the measured graph to establish whether recurrent computation
contributes to the language result. A later topology-by-trainability factorial
would also retrain these controls on the same independently rewired graphs.
Without that cross, a fixed-core result cannot establish that *anatomical*
dynamics cause the benefit. Match exposure and report wall time; do not hide
the unequal number of trainable parameters or tune one condition on test data.

Report held-out bits/byte and per-article differences with the same denominator.
The test split has already informed this research program, so these would be
exploratory extensions. Intervals crossing zero do not establish equivalence:
that would require a justified, declared equivalence margin and adequate power.
The possible conclusions remain conditional on this subset, corpus and budget.
