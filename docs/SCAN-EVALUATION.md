# Whole-partition instruction evaluation

Kuber Mehta · 11 September 2026. This is software preparation. **No official
SCAN models have been fitted or evaluated.** BabyLM and the chosen
neuron-selection comparison retain priority; the actual instruction cost pilot,
budget and protocol are still pending.

The [evaluator](../flm/scan_evaluate.py) connects the
[36-condition training coordinator](SCAN-STUDY-COORDINATOR.md) to the existing
unconstrained generation and action-sequence scorer. It addresses two practical
requirements: all registered fits must finish before test access, and a stopped
evaluation must resume without dropping difficult commands or replacing errors.

## What enters the model

The [test loader](../flm/scan_test_inputs.py) verifies the source revision,
publisher file inventory, parser version and complete test metadata frozen before
training. It hashes the raw publisher file and processed JSON records, then checks
every row against the publisher's original order, source line and multiplicity.
It rejects deduplication, reordered records, changed targets and altered counts,
including when the processed file and its recorded checksum were changed together.
Other splits' payloads and training files are not needed by this loader.

The grammar interpreter checks source integrity only. Generation receives the
command strings; reference actions enter the scorer after output has been
produced. There is no action-vocabulary mask, grammar repair or fallback answer.
The tokenizer and full output vocabulary are those of the corresponding frozen
language source. The [data note](INSTRUCTION-TRANSFER.md) explains the six action
byte IDs and the limitations of this artificial task.

## Completion gate and resumption

Before opening any test payload, the evaluator reruns the complete terminal
checkpoint audit, reconstructing all 36 conditions from code. It then holds the
training and evaluation directory leases together, verifies the study identity,
selection, source files and completion records, and loads each entire test
partition. It freezes a separate evaluation identity with the verified coverage,
policy, software sources and selected checkpoints.

Each condition restores its own terminal tensors. A saved batch contains up to
32 contiguous publisher rows; equal-length command prefixes share a model batch
within that group. The 49-token cap includes EOS, the common context is 96 tokens,
and finished examples stop independently. No reference target length sets an
example's generation budget. Recurrent/attention state resets between examples;
it carries through that example's generated tokens.

Batch records retain the complete raw token IDs, decoded actions, command prefix,
stop reason and generation time. Their identity binds exact ordered source rows,
model state, tokenizer and evaluation declaration. Cached batches are decoded and
scored again on resume. Missing interior batches, extra batches, incompatible
identities and inconsistent output records fail rather than being adopted.
Completed condition reports seal every batch checksum. A final summary is written
only after all 36 condition reports and their complete batch inventories have
been checked again. Attempts retain failure records; an incomplete evaluation has
no whole-study result.

These are reproducibility and corruption checks, not cryptographic proof that an
arbitrary supplied token sequence came from a model. Before a condition result
seals its checksums, a deliberately rewritten but internally consistent cached
prediction would require independent regeneration to detect. The runner generates
new batches directly from the restored model and checks that its parameters stay
unchanged; it does not claim to authenticate externally fabricated records.

## What the report means

The primary outcome is the entire correct action sequence **and EOS**. The report
also retains edit distance, the fraction of outputs containing invalid tokens,
empty/invalid sequence frequency, EOS termination, cap exhaustion and results by
reference action length. Invalid and unfinished outputs remain in the denominator.

Reports keep every split, initialization and seed. They compare WikiText-trained
versus original initialization within each architecture, and FLM versus GRU and
transformer within each initialization. Paired comparisons include commands only
the first model gets right, only the second gets right, both and neither. Two-seed
means are descriptive; there is no significance claim or assumption that grammar
commands are independent. A positive first-minus-second exact-match difference
favors the first condition; a negative edit-distance difference favors the first.

Language pretraining adds compute, so equal downstream updates do not equalize
lifetime compute. The three splits reuse one command universe and remain separate
experiments. Instruction composition does not establish conversation quality,
physical execution or a biological learning effect. Both improvement and failure
will be reported under the same evaluation policy.

## Verification and later use

The [loader tests](../tests/test_scan_test_inputs.py) use artificial publisher files.
The [evaluation tests](../tests/test_scan_evaluate.py) exercise invalid tokens,
empty outputs, the full 49-token cap, changed cache identities, interruptions,
completed-result corruption and late mutations before summary publication.
They also fit and evaluate all 36 small fixture conditions, preserving every
terminal checkpoint and output on resume. Those fixtures use six training updates,
tiny models and a four-token generation cap; they are not official transfer
measurements or a throughput pilot. The full-cap check is a separate scripted
runtime fixture, never exported as a model result.

```powershell
python -m unittest discover -s tests -p test_scan_test_inputs.py -v
python -m unittest discover -s tests -p test_scan_evaluate.py -v
# Only after priority studies, measured costs, a frozen protocol and all fits:
python -m flm.scan_evaluate
```

The evaluator has no command-line option for a shortened condition inventory,
test subset, substitute checkpoint or output repair. The official study remains
unstarted; a working evaluation command alone is not a benchmark result.

The [dated verification record](../reports/scan-runtime/evaluation-preparation.json)
records **31 passed checks**: six loader, thirteen evaluation and twelve training
coordinator checks. Running the evaluator against the actual repository refused
the absent frozen study before calling the test loader or creating any official
evaluation artifacts. All benchmark fits, costs and scores remain pending.

A subsequent [inventory verification](../reports/scan-runtime/evaluation-inventory-verification.json)
tightens the final check against a batch added after a condition has completed.
It tests that failure and reruns the complete 36-condition fixture evaluation
and cached resumption on the revised evaluator. The earlier 31-check record
retains its original source commit; the addendum identifies the revised code.
