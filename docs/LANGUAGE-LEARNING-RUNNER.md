# Resumable training for alternative language learning rules

The [runner](../flm/language_learning_train.py) fits four conditions using supplied
tokenized training documents: BPTT, fixed core, forward eligibility and eligibility
without history. It does not acquire data, launch itself, choose an official
budget, select a validation checkpoint or evaluate test data. Only small
artificial software fixtures have been fitted with it.

The subsequent [verified input adapter](LANGUAGE-LEARNING-INPUTS.md) now binds
the actual BabyLM training sources, tokenizer, cache and original graph. It
prepares all eight rule/seed conditions and passes separate full-size synthetic
update checks. Official corpus fits and the evaluation protocol remain pending.

All conditions share ordinary FLM forward dynamics. Eligibility changes the
gradient using the [declared diagonal temporal approximation](LANGUAGE-ELIGIBILITY-KERNEL.md).
History removal discards old derivatives while preserving forward state.
The fixed-core condition freezes `input.weight`, `input.bias`, `edge_log_gain`,
`alpha_logit`, `beta_logit` and `recurrent_logit`. It trains the embedding and
classifier, including pooling normalization, with ordinary backpropagation
through the fixed recurrence. It is neither the sensory study's readout-only
reservoir nor the completed language study's fixed-dynamics condition, which
left the input projection trainable. At the current 600,003-parameter language
configuration this definition freezes 177,507 entries and trains 422,496.

## Common fitting contract

`Settings` explicitly supplies the update count, batch and sequence dimensions,
unscored warm-up length, optimizer schedule, clipping, checkpoint interval, seed,
CPU thread count and whether to score boundary tokens. The default excludes
target IDs 0 and 1. This differs from the historical language trainer's training
loss, which included boundary targets after warm-up. Therefore the future
comparison needs fresh, matched BPTT fits; old language scores cannot substitute
for them. All conditions start from zero recurrent state per sampled window.

The existing window sampler chooses eligible documents by valid-window count and
then a uniform window start. Documents with length at most `sequence + 1` are
excluded, exactly as in that sampler. The declaration records the entire
supplied ordered document inventory and eligible count. Empty scored masks fail
without resampling, preventing condition-dependent changes to the data stream.

One window produces one mean scored-token objective, one gradient clip and one
AdamW update. The cosine schedule and optimizer warm-up use the full declared
update budget even when execution stops at an intermediate checkpoint. The
forward edge clamp remains in FLM's model equations; the runner does not add a
post-update projection of those parameters.

## Binding and recovery

Supply a fresh copy of the identical initial model when starting or resuming.
The declaration binds its complete tensor hash, graph/pooling buffers, model
configuration, method, ordered training-document content, tokenizer identity and
byte-length table, settings, software versions and numerical source hashes.
An explicit nonempty study/data binding is required. Actual official split and
license verification must happen in the later dataset-specific launcher; this
generic function cannot establish that caller-supplied documents are a valid
training partition.

An OS-held directory lock excludes concurrent writers. Checkpoints are written
atomically, then committed by a separate atomic JSON record with their checksum.
A payload without that final record is ineligible to resume. On restoration the
runner checks all model tensors, unchanged graph and frozen core values, optimizer
parameter order, every moment's shape/dtype/finiteness and update counter, training
history and the full learning-rate schedule. It replays the sampled token stream
to verify its digest, NumPy RNG and exact presented/scored token and UTF-8 byte
exposure. Torch RNG is also restored. Eligibility starts fresh after each update
and need not persist across saved window boundaries.

The endpoint record means only that all requested training updates completed.
It is explicitly not validation selection or permission for official test
evaluation. A future all-condition selection gate must precede that evaluation.

## Verification and next steps

Seven tests exercise uninterrupted versus resumed fitting across all four
conditions; they compare complete payloads, including weights, optimizer, RNG,
sample digest, exposure and history. Other checks cover the fixed-core inventory,
an independently computed masked BPTT/AdamW update, a crash between checkpoint and
commit record, changed data/settings/initialization, six kinds of payload
tampering even with an updated file checksum, empty masks and live writer locks.

```powershell
python -m unittest discover -s tests -p test_language_learning_train.py -v
```

The [cost pilot](LANGUAGE-LEARNING-TIMING.md) now calls this exact update path
and is prepared for full batch/sequence measurements after the priority queue.
Its current full-window evidence covers initialization and sampling only.

These are tiny one-thread fixtures, not training throughput or four-thread
reproducibility measurements. Before official fits, finish the current BabyLM
queue, use the verified training-only adapter, measure disposable full-window costs,
freeze a common budget and selection protocol, and implement the all-condition
evaluation gate. The new selection-method study is a separate experimental axis;
do not silently change graphs while comparing learning rules.
