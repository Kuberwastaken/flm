# Validation selection for alternative learning rules

The [training runner](LANGUAGE-LEARNING-RUNNER.md) saves resumable endpoints.
The new [validation selector](../flm/language_learning_validation.py) checks a
completed run and chooses its lowest-validation-loss checkpoint, with the
earliest checkpoint winning an exact tie. This applies the same forward
evaluation to BPTT, fixed-core, eligibility and no-history training.

**This is a tested per-run component, not a frozen official experiment.**
The cost pilot has not run, the training budget remains undecided, and the
official study-wide identity has not been frozen. The
[coordinator and all-condition selection gate](LANGUAGE-LEARNING-STUDY.md) are
now implemented and tested; the official test scorer remains to be implemented.
No official alternative-learning checkpoint has been trained or selected.

## The validation panel

The selector's input loader pins the existing BabyLM validation cache and
score-independent panel by SHA-256. It reads the first 1,025 IDs of each of 48
selected blocks: one starting input and up to 1,024 target positions. The
original panel selected eight blocks per component by hashed stable IDs, without
model losses. Reusing it avoids choosing examples after observing a learning
rule's performance. These block prefixes are not the full validation split.

The [actual-input preflight](../reports/language-eligibility/validation-preflight.json)
verifies the following denominators, without constructing a model or calculating
any likelihoods:

| Component | Blocks | Scored tokens | Scored UTF-8 bytes |
|---|---:|---:|---:|
| BNC spoken | 8 | 8,192 | 27,711 |
| CHILDES | 8 | 8,192 | 19,643 |
| Project Gutenberg | 8 | 8,192 | 27,659 |
| OpenSubtitles | 8 | 8,192 | 23,295 |
| Simple Wikipedia | 8 | 8,192 | 25,452 |
| Switchboard | 8 | 8,192 | 22,015 |

The loader opens only the validation cache and panel, using the train-fitted
tokenizer supplied by the caller. It verifies the cache payloads through the
existing mmap reader, copies the selected prefixes into read-only arrays, and
binds their exact ordered IDs, token bytes and byte-length table. No test payload
is opened. Disjoint block IDs do not by themselves establish content deduplication
between corpus partitions; this component makes no such claim.

## How a checkpoint becomes eligible

The caller supplies the fresh initial model, original training documents and
bindings, explicit training settings and method. These must reproduce the
training declaration exactly. The selector requires the completed endpoint and
every declared checkpoint at the full save cadence. It holds the same directory
writer lock as training while selecting.

Before calculating any validation loss, it checks every marker and checkpoint
checksum and fully restores every payload. This verifies model tensors, immutable
graph buffers and frozen parameters, optimizer groups and moments, update counts,
training history, exact sampled-token digest, RNG and token/byte exposure. The
final restored exposure must agree with the completion record. An interrupted
or changed run cannot enter selection merely because a final weights file exists.

Each checkpoint then uses the ordinary FLM forward computation, regardless of
its training gradient rule. Fast and slow state reset at each block and carry
across scoring chunks within that block. Targets with IDs below 2 are excluded;
all text targets are scored, without the training window's warmup mask. Aggregate
bits per byte use summed negative log likelihood divided by summed UTF-8 bytes,
not an unweighted mean over blocks. The selection record includes each block's
loss and denominator and every checkpoint's identity.

The selector checks denominators and arithmetic independently of the evaluator,
requires every checkpoint exactly once, and chooses the earliest exact minimum.
It rechecks the panel and training inventory before atomically writing
`validation-selection.json` in the run directory. The supplied initial model,
global Torch RNG and thread setting are preserved.

Reopening a saved selection restores and audits all training payloads again,
then validates the stored score identities, coverage, arithmetic and selection
rule. It does not recompute cached likelihoods. This cache check is distinct from
independent inference reproduction; the future study-wide freeze must bind the
selection records' checksums before test access.

## Verification and remaining work

Eight artificial-fixture tests cover all four learning rules, whole-block versus
chunked likelihood calculations, exact byte normalization, earliest ties,
incomplete or altered endpoints, rehashed invalid optimizer moments, changed
panels, overlapping IDs, inconsistent cached scores, failure cleanup and restored
caller state. None trains on an acquired corpus or selects a real checkpoint.

```powershell
python -m unittest discover -s tests -p test_language_learning_validation.py -v
python -m scripts.language_learning_validation_preflight --output reports/language-eligibility/another-dated-validation-preflight.json
```

The [cost preparation](LANGUAGE-LEARNING-TIMING.md) remains separate. After actual
costs are measured, use the coordinator to freeze the complete eight-condition
identity and exposure budget before fitting. The future test scorer must call
the all-condition gate before official test evaluation. Existing BabyLM, selection, sensory and physical
protocols remain unchanged.
