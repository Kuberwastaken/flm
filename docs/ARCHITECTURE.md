# FLM 0.1 architecture specification

This specification precedes training. Changes and results are versioned separately.

## Representation

Input is normalized transcript text encoded as UTF-8 bytes (0–255), plus BOS=256 and EOS=257. All embeddings initialize from scratch. Lowercasing and whitespace normalization are declared data transformations; the vocabulary itself is fixed without learning from evaluation data.

The initial published experiment uses a compact induced central-brain subgraph, selected by contact strength before inspecting language results. Source IDs, cell types, positions, connections and discarded boundary behavior are recorded. Whole-CNS acquisition and the compact model are distinct assets.

## Recurrent computation

For each byte embedding `e_t`, compute:

```
drive = input(e_t) + recurrent_gain * W @ h + bias
h_new = (1 - alpha) * h + alpha * tanh(drive)
slow_new = (1 - beta) * slow + beta * h_new
features = concatenate(pool(h_new), pool(slow_new))
logits = readout(features)
```

`W` is sparse and follows only observed directed edges. Its initial magnitude is log(1+contacts), normalized by absolute incoming sum. Trainable positive edge gains preserve source sign. `alpha` and `beta` are bounded per-neuron parameters; beta has a slower range. Model state has no access to future tokens. All topology variants use the same equations and interface sizes.

The anatomical subgraph and balanced type-ordered pooling form the prior. Input and readout are learned artificial interfaces. No direct embedding-to-logit bypass, external language model, or canned answer generator appears in the primary path.

## Learning

Persistent weights learn through teacher-forced next-byte cross-entropy and truncated backpropagation. Chunks do not cross transcript-document boundaries; state is reset or explicitly continued only within the same document. Training must support resumable optimizer/RNG state and capped runs.

Browser personalization initially updates a separate low-rank/readout correction using actual recurrent features and the next byte observed in user-provided text. It never changes the bundled checkpoint. Corrections are saved only with user action and can be reset/exported. This is supervised local adaptation, not a claim to implement biological synaptic learning. Evaluate before/after loss on separate adaptation and probe text.

## Experiments

Compare full proposed core, no slow state, recurrence disabled and a genuinely rewired directed graph preserving degrees and source-sign constraints. Also report trivial language baselines. A lower training loss alone does not support a quality claim. A compact model does not establish whole-brain scaling, and a single-seed result does not establish general topology superiority.
