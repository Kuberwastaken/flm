# Training inputs for the learning-rule comparison

The [input adapter](language-learning/language_learning_inputs.py) now connects the
[alternate-rule trainer](language-learning-runner.md) to verified BabyLM 2026
`train-10m` data. It prepares BPTT, fixed core, forward eligibility and eligibility
without history at initialization seeds 42 and 43. **These eight preparations
are not eight completed language fits.** No training budget, validation selection
or whole-study test gate is defined by this adapter.

## Verified bytes, shared inputs

The [dated preflight](language-learning/input-preflight.json)
rechecks all six raw training components against the pinned publisher revision,
acquisition SHA-256 records and Git blob or LFS identity. It verifies the
train-fitted tokenizer and token-cache identities, then decodes every block and
compares it with the exact source byte range. Every source byte is covered once:
**3,351 blocks, 18,591,514 text tokens and 54,399,840 UTF-8 bytes**. Two boundary
IDs per block account for the larger cache-token count reported elsewhere.

The loader rejects missing or duplicate components, changed source bytes,
incorrect source ranges, repeated block identities, malformed boundary IDs,
incorrect decoded text and changed aggregate counts. Shared inventory metadata
includes other dataset partitions, but their raw text and token payloads are not
opened. Source URLs, hashes and the existing component-rights qualification are
retained. No raw corpus is added to the repository or site.

The graph is explicitly pinned to the original ranked 1,024-neuron subset. The
new KC-centered selections belong to a separate experimental axis; changing
them here would confound the learning-rule comparison. The adapter acquires no
new graph and loads no trained language checkpoint.

## Identical starting tensors, different credit assignment

For each seed, all four rules receive exactly identical initial model tensors,
including graph and pooling buffers. Fresh array copies prevent one prepared
model from mutating the shared graph or another condition. The seed changes
between the two groups. Initialization preserves the caller's Torch RNG.

BPTT, eligibility and no-history each expose all 600,003 parameters to their
optimizer. Fixed core exposes 422,496 and freezes the 177,507 input/core entries
defined by the existing runner. Its embedding and classifier still learn through
the fixed recurrence. This differs from the previously completed fixed-dynamics
control, which left the input projection trainable.

The preflight checks that all eight preparations are accepted by the generic
trainer's declaration machinery with the same ordered training-document digest.
Separately, each full-size configuration passes two tiny optimizer updates on
synthetic token IDs, using batch 1, three input tokens and one CPU thread. The
fixed core remains unchanged. The real corpus is not used for those updates;
their dimensions are software fixtures, not a selected language-study budget.
No checkpoints, language scores or throughput estimates are produced.

Eligibility keeps the ordinary FLM forward dynamics but approximates temporal
credit assignment by omitting paths between different neurons. It still uses
autograd for the current readout and AdamW for parameter updates. This is not a
validated biological learning mechanism or a capability exclusive to fly wiring.
The no-history control removes stored derivatives while retaining forward memory.
See the [kernel specification](language-eligibility-kernel.md).

## Reproduction and remaining work

```powershell
python -m unittest discover -s tests -p test_language_learning_inputs.py -v
python -m scripts.language_learning_input_preflight --output reports/language-eligibility/another-dated-input-preflight.json
```

Six unit tests cover byte reconstruction, source and token tampering, missing
held-out payloads, initialization matching, graph-storage isolation and invalid
conditions. The [preflight script](language-learning/language_learning_input_preflight.py)
uses the actual acquired inputs and refuses to overwrite its output.

The [full-window cost pilot](language-learning-timing.md) is now implemented;
its declarations and actual sampling pass for all eight conditions, with no
gradient updates or timing observations in that preflight.

Before official fits, finish the priority queue, measure full-window costs for
each rule, freeze a common budget and all-condition validation/test protocol,
then train fresh matched BPTT controls. The runner's default training mask
excludes boundary targets, unlike the historical language trainer; earlier BPTT
scores therefore cannot substitute for those controls. No existing language,
selection, sensory or physical experiment is changed by this preparation.
