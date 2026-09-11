# Held-out evaluation for neuron-selection groups

The [selection evaluator](../flm/selection_language_test.py) implements complete
held-out likelihood scoring and the comparison plan below. **No official
selection-language test score exists.** The measured cost pilot, chosen groups,
common training budget, written protocol and actual fits remain pending.

The scoring policy and evaluator source hashes are now included in the
[coordinator's](SELECTION-LANGUAGE-COORDINATOR.md) frozen identity before fitting.
They cannot be changed afterward while retaining the same study identity. The
[dated readiness record](../reports/selection-language/evaluation-verified-preparation.json)
checks the released catalog, comparison count and pending gates without opening
corpus payloads or executing a model. Earlier preparation records retain their
original source hashes and implementation status.

## Score the same complete test blocks

The evaluator first requires the complete chosen-group inventory. It audits
every condition's completed training and validation-selected checkpoints before
opening any test-token payload. A missing or damaged last condition prevents
test access for the whole experiment. Test likelihoods never select checkpoints.

The frozen tokenization card and test manifest bind the actual cache files.
Payload checks cover hashes, unique block identities, component coverage, token
boundaries, byte counts and the fixed overlap-filter eligibility flags. Evaluation
uses owned read-only copies of the token arrays and releases their file mappings.

Each model starts with zero recurrent state at a block boundary and carries its
state across 96-token chunks within that block. Eight blocks are evaluated per
batch using the shared likelihood kernel on four CPU threads. All actual text
targets are scored, with no evaluation warmup; inserted BOS/EOS targets are
excluded. Padded suffixes have no scored targets. Float64 reductions accumulate
the float32 token losses into complete-block negative log likelihoods.

Results include official and fixed overlap-filtered scores, per source component
and for the whole corpus. The metric is total negative log likelihood divided by
total UTF-8 bytes and `log(2)`, rather than an unweighted average of block BPBs.
An empty filtered component has a null result and an explicit reason where needed.

Every selected model is restored from its actual checkpoint using the complete
training declaration, optimizer, sampler and exposure checks. Atomic batch files
permit interrupted evaluation to resume. Cached records must match the frozen
identity and block inventory; arithmetic, denominators and completion hashes are
rechecked. Test payloads and the complete study context are verified again before
the final all-condition summary is written. Caching avoids repeating inference;
it does not independently recompute previously saved likelihoods.

## Separate selection from retained topology

For each chosen group, training seed and corpus subset, the report retains:

| Question | Family contrasts | Individual contrasts |
|---|---:|---:|
| Candidate versus contact-ranked selection | 1 | 1 |
| Candidate versus three uniform selections | 1, using their mean loss | 3 |
| Candidate versus three stratified selections | 1, using their mean loss | 3 |
| Each of eight selections versus its three rewires | 8, using each rewired family's mean loss | 24 |
| **Total** | **11** | **31** |

That is 42 comparison records per training seed and corpus subset, or 168 per
chosen group across seeds 42/43 and the two subsets. The single ranked contrast
appears in both tables intentionally; it is the same estimate, not extra evidence.

Family aggregation takes the arithmetic mean of the three **per-block negative
log likelihoods**, keeping the common block byte/token denominator. It is not
an ensemble that averages next-token probabilities. All three graph outcomes
remain available individually. Training-seed summaries pair the same comparisons
across seeds 42/43 and report their mean difference and sample standard deviation.

The difference is first minus second in bits per byte: a negative value favors
the first model or family. Candidate-versus-selector comparisons hold neuron
count fixed but can differ in edge counts and parameter allocations. A measured
subset versus its own rewires holds those allocations fixed. The report preserves
each run's graph, selection rule, counts and source checksum so those two questions
cannot be silently conflated.

## What the intervals do and do not establish

Each comparison uses 10,000 paired bootstrap replicates, seed 31415, resampling
blocks within their source component. The family interval conditions on those
particular three graphs as well as on the fitted checkpoints. It does not
resample graphs or create more training replications from rewires.

The intervals are descriptive and unadjusted for the many comparisons. Adjacent
artificial blocks may share an unknown source document. Two training seeds and
three graph samples cannot establish broad robustness. Report every comparison;
do not select a favorable candidate group, graph seed or component after seeing
test scores and present it as a preregistered result.

These are language likelihood measurements on truncated operational candidates.
They do not measure conversation ability, fruit preference, physical behavior or
intact mushroom-body computation. No result here would settle whether biological
wiring is useful in general.

## Verification and eventual command

The evaluator tests use artificial graphs and text. They check the exact
comparison inventory, loss averaging, paired seed summaries, empty filtered
subsets, unequal coverage and missing results. The complete-group test trains
all 64 tiny conditions, interrupts held-out scoring after its first saved batch,
resumes the other 127 batches, and compares every condition's scored blocks with
direct whole-block likelihoods. It then verifies cached replay and rejection of
changed policies, evaluator source, the final selected checkpoint, token payloads,
cached arithmetic and completed result files. These are software fixtures, not
acquired-corpus timing or language results.

```powershell
python -m unittest discover -s tests -p test_selection_language_test.py -v
# After registration, every fit, and frozen whole-group validation selection:
python -m flm.selection_language_test
```

Official output belongs in `reports/selection-language/test-summary.json`, with
per-condition blocks in `reports/selection-language/test/` and resumable batches
under `runs/selection-language-evaluation-v1/`. Those official results are absent.
