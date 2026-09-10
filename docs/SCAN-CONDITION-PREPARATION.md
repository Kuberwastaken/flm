# Matched instruction-condition preparation

The [condition layer](../flm/scan_conditions.py) connects the verified official
training loader and original language sources to the resumable fitting helper.
All [36 combinations were prepared](../reports/scan-runtime/condition-preflight.json)
against real files without fitting or predicting SCAN examples. This is not a
frozen downstream study or a result about transfer.

The inventory crosses three official splits, FLM/GRU/transformer, seeds 42/43,
and original initialization versus the selected WikiText weights. Each condition
has a unique label. Both initial and pretrained conditions retain the same
train-fitted WikiText tokenizer; the initial condition has no learned language
weights, but it shares that preprocessing prior. Pretrained conditions have
already consumed language-training data and compute.

`prepare_condition(root, condition)` reloads the official training rows and
verifies the six-source audit, numerical source hashes, language selection,
tokenizer, graph and actual checkpoint tensors. It reconstructs the original
initial state and checks it against the historical audit. The returned model
contains either those initial tensors or the selected language tensors; no
language optimizer state is inherited.

Before calling `scan_train.fit`, use `prepared.training_binding(settings)`.
This checks that the model, ordered rows and tokenizer have not changed and
requires the sampling seed to match the condition's seed. The returned binding
records all data/source identities and is copied into the fitting declaration.
Prepare a fresh original source again before resuming; the fitter restores the
downstream checkpoint into that verified architecture. Do not reuse the mutated
model from a partially completed call as its declared initialization.

Six tests cover the inventory, invalid conditions, stale source audits, changed
inputs, sampler mismatch and integration with checkpoint resume. The three
small architecture fixtures reproduce uninterrupted final tensors, loss history,
sampled rows and exposure after resumption. These tests use six-update artificial
examples, not official SCAN training or a multithreaded reproduction guarantee.

The [train-only timing pilot](SCAN-TIMING-PILOT.md) now has a tested command,
but its full-size measurements await the priority queues. Remaining work is that
measurement, the fixed optimizer/exposure budget,
immutable whole-study declaration, serial execution and an all-condition gate
before official test generation. The preparation API intentionally sets no
training budget and runs no scheduler. Keep the language computation queue and
registered BabyLM comparison ahead of these new fits. See the
[experimental design](INSTRUCTION-TRANSFER.md) for interpretation and limits.

```sh
python -m unittest discover -s tests -p test_scan_conditions.py -v
```
