# FLM 0.1 architecture specification

The initial specification preceded training. This document now records the implemented 0.1 core; results and comparison protocols are versioned separately.

## Representation

Input is normalized transcript text encoded as UTF-8 bytes (0–255), plus BOS=256 and EOS=257. All embeddings initialize from scratch. Lowercasing and whitespace normalization are declared data transformations; the vocabulary itself is fixed without learning from evaluation data.

The initial published experiment uses a compact induced central-brain subgraph, selected by contact strength before inspecting language results. Source IDs, cell types, positions, connections and discarded boundary behavior are recorded. Whole-CNS acquisition and the compact model are distinct assets.

## Recurrent computation

For each byte embedding `e_t`, compute:

```
drive = input_linear(e_t) + recurrent_gain * W @ h
h_new = (1 - alpha) * h + alpha * tanh(drive)
slow_new = (1 - beta) * slow + beta * h_new
features = concatenate(pool(h_new), pool(slow_new))
logits = readout(layer_norm(features))
```

`W` is sparse and follows only observed directed edges. Its initial magnitude is log(1+contacts), normalized by absolute incoming sum. Trainable positive edge gains preserve source sign. `alpha` and `beta` are bounded per-neuron parameters; beta has a slower range. Model state has no access to future tokens. All topology variants use the same equations and interface sizes.

`input_linear` includes one bias. Layer normalization uses epsilon 1e-5. Edge gains are `exp(clamp(theta, -3, 3))`, followed by renormalization of the absolute incoming row sum. The recurrent scalar is `0.05 + 2.95 sigmoid(g)`. Fast update rates are `0.05 + 0.90 sigmoid(a)`; slow rates are `0.002 + 0.098 sigmoid(b)`. These are discrete byte-update parameters, not milliseconds or measured membrane constants.

At 1,024 neurons the Python backend uses a dense matrix representation of the sparse topology because this was faster on the available CPU. The browser uses destination-major CSR. Neither implementation is an event-driven spiking simulator. Sparse topology alone does not establish lower energy consumption.

The anatomical subgraph and balanced type-ordered pooling form the prior. Input and readout are learned artificial interfaces. No direct embedding-to-logit bypass, external language model, or canned answer generator appears in the primary path.

## Learning

Persistent weights learn through teacher-forced next-byte cross-entropy and truncated backpropagation. Chunks do not cross transcript-document boundaries; state is reset or explicitly continued only within the same document. Training must support resumable optimizer/RNG state and capped runs.

Browser personalization updates a separate full readout correction (258 by 256 coefficients plus bias) using actual normalized recurrent features and the next byte observed in user-provided text. It never changes the bundled checkpoint. Corrections are saved only with user action and can be reset/exported. This is supervised local adaptation, not a claim to implement biological synaptic learning. Evaluate before/after loss on separate adaptation and probe text. For fixed input, the adapter cannot change fast/slow activity; during free generation, changed predictions can change the subsequent input trajectory.

## Experiments

Compare full proposed core, no slow state, recurrence disabled and a genuinely rewired directed graph preserving degrees and source-sign constraints. Also report trivial language baselines. A lower training loss alone does not support a quality claim. A compact model does not establish whole-brain scaling, and a single-seed result does not establish general topology superiority.
