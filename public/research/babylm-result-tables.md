# BabyLM result tables

The [exporter](babylm-result-tables/babylm_result_tables.py) converts the complete saved
BabyLM likelihood experiment into JSON and CSV tables for figures, papers and
the browser. **It has passed artificial-fixture tests; it has not exported an
official BabyLM result.** The twelve-fit evaluation remains pending.

The exporter is separate from the [armed handoff](babylm-handoff.md). It does not
alter that running supervisor or launch inference, generation or timing. Run it
after the declared evaluation has completed:

```powershell
python -m unittest discover -s tests -p test_babylm_result_tables.py -v
python scripts/babylm_result_tables.py --output reports/babylm/tables-v1
```

The destination must be fresh. Missing selection, summary or any of the twelve
per-run test files stops export before reading test metadata or creating output.
The test-token payload is never opened. Source and input hashes are checked
again after calculation; the output manifest is written last. A filesystem
failure can leave an incomplete destination, which must be inspected rather
than overwritten.

## What the tables preserve

- `scores.csv`: all twelve runs × seven component labels (six sources plus the
  pooled mixture) × two analyses. The 168 rows retain negative log likelihood,
  exact byte/token/block denominators, BPB, same-tokenizer perplexity, exclusions,
  parameter count, selected update/hash and final training exposure. An empty
  filtered component is marked unavailable, with blank score fields, not zero.
- `seed_aggregates.csv`: both training-seed BPBs, their arithmetic mean and sample
  standard deviation, separately for every available component and analysis.
  This SD is not a confidence interval. A wholly excluded component has no
  filtered mean row.
- `paired_comparisons.csv`: all sixteen declared FLM-minus-comparator contrasts,
  preserving scale, training seed, analysis and the evaluator's interval method.
  Positive differences favor the comparator; lower BPB is better.
- `tables.json` and `manifest.json`: the same tables with interpretation notes,
  input/evaluator/exporter hashes, and hashes and sizes of the four result files.

## Checks and their limits

The exporter independently adds saved block likelihoods and byte denominators.
It does not average block or component BPBs to obtain the pooled score. Every
run must cover the same complete prepared block inventory and the recorded
component, text counts and overlap flags. Official and overlap-filtered scores
stay separate. Component arithmetic, exclusions, seed aggregates and paired
point-estimate signs must agree with the saved evaluator summary.

Interval endpoints are copied from the bound evaluator output, with finite-value,
ordering and declared-method checks. **This exporter does not rerun the 10,000
bootstrap draws or certify their numerical endpoints independently.** It also
does not restore checkpoints or reproduce model likelihoods. Those responsibilities
remain with the [evaluation implementation](babylm-evaluation.md). Artificial
blocks may share unknown documents, and two seeds give limited training
uncertainty. Neither pooled nor component likelihood establishes conversation,
instruction-following or biological capability.

The [ten tests](babylm-result-tables/test_babylm_result_tables.py) use twelve artificial runs,
six source labels, deliberately unequal block sizes and one wholly excluded
source. They check correct weighting, nulls, two-seed SD, contrast direction,
complete inventories, mutations and fresh-output behavior. The
[preparation record](babylm-result-tables/preparation.json) retains
their scope and the actual still-closed export gate. It is not a benchmark result.
