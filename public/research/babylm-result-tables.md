# BabyLM result tables

The [exporter](babylm-result-tables/babylm_result_tables.py) converts the complete saved
BabyLM likelihood experiment into JSON and CSV tables for figures, papers and
the browser. **The complete twelve-fit experiment was exported on 12 September
2026:** 168 score rows, 84 seed aggregates and 16 paired comparisons. See the
[findings](babylm-findings.md) and [table manifest](babylm-results/tables/manifest.json).

The exporter is separate from the [completed handoff](babylm-handoff.md). It does
not launch inference, generation or timing. The following commands document
the export; choose a new destination for another reproduction:

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

## Computational figures

The [figure builder](babylm-result-tables/babylm_result_figures.py) calls the same complete
artifact/arithmetic gate before importing Matplotlib or creating its output.
It produces two PNG/SVG pairs:

- Pooled held-out loss: four panels separate 10M/100M training corpora and
  official/overlap-filtered analyses. Every panel uses the same horizontal
  scale. The two training seeds and their descriptive mean have distinct
  markers; no confidence interval is implied. Scored bytes and blocks are shown.
- Source-specific differences: all six components, both corpus sizes and both
  analyses, pairing FLM with each comparator at the same training seed. A shared
  symmetric axis retains zero. Negative favors FLM; positive favors its
  comparator. Both seeds are shown, without component confidence intervals.
  A wholly excluded source is labeled, with no zero-valued point.

```powershell
python -m unittest discover -s tests -p test_babylm_result_figures.py -v
python scripts/babylm_result_figures.py --output reports/babylm/figures-v1
```

The fresh destination contains the four images, the complete underlying checked
tables and component differences in `figure-data.json`, and a manifest written
only after all image files exist and the inputs still match their hashes.
Inspect any partial destination after a rendering failure; it is not a completed
figure release. The builder is not attached to the live training supervisor.

The [seven figure tests](babylm-result-tables/test_babylm_result_figures.py) cover pairing,
missing data, denominators, incomplete-study gating, existing outputs, changed
inputs and shortened rendering. The [layout review](babylm-result-tables/figures-preparation.json)
records two inspected synthetic PNGs and their matching SVGs. Those private
fixtures visibly say they are not benchmarks and are not published as results.
The actual-value pooled and component PNGs have now also been generated and
visually reviewed at 1920 × 1440, with readable labels and no clipping. The
[completed evidence record](babylm-results/completed-evidence-verification.json)
records that review; the fixture review remains separate historical evidence.
