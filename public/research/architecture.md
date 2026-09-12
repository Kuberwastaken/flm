# FLM architecture specification

The current browser release uses the **0.2 lexical architecture** for the completed WikiText and BabyLM models. The default is BabyLM 100M FLM, seed 43, chosen by fixed-panel validation loss. The earlier 0.1 AMI byte model remains available as a separate legacy checkpoint. The active 487/540-neuron selection study is not a validated browser release.

FLM is trained from scratch through a connectome-derived recurrent core. Its input and output interfaces are learned artificial components; it is neither a pretrained transformer nor an intact simulated fly brain.

## Current lexical interface (0.2)

The completed WikiText and BabyLM experiments use the recurrent core below with 4,096 train-only byte-BPE IDs (BOS=0, EOS=1). A token can contain multiple bytes, so the discrete update rates now apply per token, not per byte or millisecond. A 96-dimensional embedding is shared by the input and output interfaces:

```
e_t = E[x_t]
h_t, s_t = recurrent_core(input_linear(e_t), h_previous, s_previous)
f_t = layer_norm(concatenate(pool(h_t), pool(s_t)))
z_t = projection_256_to_96(f_t)
logits_t = E @ z_t + output_bias
```

There is no attention or direct token-to-logit bypass. The shared embedding has 393,216 coefficients; the total model has 600,003 trainable parameters. A small GRU (595,408) and two-layer RoPE transformer (607,468) share this lexical interface and tokenizer. Full settings are registered in [WikiText](https://github.com/Kuberwastaken/flm/blob/main/docs/WIKITEXT-PROTOCOL.md) and [BabyLM](https://github.com/Kuberwastaken/flm/blob/main/docs/BABYLM-PROTOCOL.md); each corpus has its own train-only tokenizer.

The lexical browser adapter adds a 4,096 by 96 correction plus output bias. It uses `z_t` as its observed feature and changes neither the original tied embedding nor the recurrent core. Float32 corrections are packed as little-endian base64 for checkpoint-specific storage; legacy array adapters remain readable. Scoring sums next-token negative log likelihood and divides by the exact UTF-8 byte lengths of target pieces, excluding boundary IDs. This is canonical-token-sequence codelength per byte, not a published word-token WikiText perplexity.

## Recurrent computation

For each token embedding `e_t`, compute:

```
drive = input_linear(e_t) + recurrent_gain * W @ h
h_new = (1 - alpha) * h + alpha * tanh(drive)
slow_new = (1 - beta) * slow + beta * h_new
features = concatenate(pool(h_new), pool(slow_new))
logits = readout(layer_norm(features))
```

`W` is sparse and follows only observed directed edges. Its initial magnitude is log(1+contacts), normalized by absolute incoming sum. Trainable positive edge gains preserve source sign. `alpha` and `beta` are bounded per-neuron parameters; beta has a slower range. Model state has no access to future tokens. All topology variants use the same equations and interface sizes.

`input_linear` includes one bias. Layer normalization uses epsilon 1e-5. Edge gains are `exp(clamp(theta, -3, 3))`, followed by renormalization of the absolute incoming row sum. The recurrent scalar is `0.05 + 2.95 sigmoid(g)`. Fast update rates are `0.05 + 0.90 sigmoid(a)`; slow rates are `0.002 + 0.098 sigmoid(b)`. These are discrete token-update parameters, not milliseconds or measured membrane constants.

At 1,024 neurons the Python backend uses a dense matrix representation of the sparse topology because this was faster on the available CPU. The browser uses destination-major CSR. Neither implementation is an event-driven spiking simulator. Sparse topology alone does not establish lower energy consumption.

The anatomical subgraph and balanced type-ordered pooling form the prior. Input and readout are learned artificial interfaces. No direct embedding-to-logit bypass, external language model, or canned answer generator appears in the primary path.

## Learning and evaluation

The lexical models learn through teacher-forced next-token cross-entropy and truncated backpropagation through time. Training windows respect the declared document/block boundaries and warmup masks; the pinned protocols specify sampling and state handling. The shared input/output embedding, projections, edge gains and update rates learn together. The fixed-location graph is an architectural prior, not a biological learning rule.

Completed experiments compare the measured core with retrained rewires, no-slow, fixed-dynamics, no-lateral and no-temporal controls. These are distinct interventions with different effective parameter allocations; they are not an additive decomposition. See [computation results](https://github.com/Kuberwastaken/flm/blob/main/docs/LANGUAGE-CORE-RESULTS.md), [the focused synthesis](https://github.com/Kuberwastaken/flm/blob/main/docs/PREDICTIVE-COMPUTATION.md) and [current study status](https://github.com/Kuberwastaken/flm/blob/main/docs/PLAN.md).

## Legacy AMI byte model (0.1)

The earlier AMI model uses normalized UTF-8 bytes (0-255), BOS=256 and EOS=257. Its vocabulary has 258 IDs. Persistent weights learn through teacher-forced next-byte cross-entropy and truncated backpropagation. Chunks do not cross transcript-document boundaries; state is reset or explicitly continued only within the same document. Training must support resumable optimizer/RNG state and capped runs.

Browser personalization updates a separate full readout correction (258 by 256 coefficients plus bias) using actual normalized recurrent features and the next byte observed in user-provided text. It never changes the bundled checkpoint. Corrections are saved only with user action and can be reset/exported. This is supervised local adaptation, not a claim to implement biological synaptic learning. Evaluate before/after loss on separate adaptation and probe text. For fixed input, the adapter cannot change fast/slow activity; during free generation, changed predictions can change the subsequent input trajectory.
