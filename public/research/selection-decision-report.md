# Reproducing the anatomical-prior allocation decision

Kuber Mehta · 12 September 2026. Reporting infrastructure prepared while the frozen selection study is training. No selection-language held-out result is available in this note.

The [dated rule](anatomical-prior-decision.md) defines the scientific interpretation and research allocation. The [report script](selection-decision-report.py) applies it mechanically to the completed outputs. It pins the original rule's SHA-256 and writes a separate derived record; it cannot rewrite the rule, adjust its margin, pick the best group, or launch another study.

Run from the repository root:

```sh
python scripts/selection_decision_report.py
```

Until both the complete checkpoint selection and final held-out summary exist, this prints **pending** without opening any partial held-out results. This is a snapshot, not a training progress monitor. Keep the original train/select/test supervisor running unchanged.

After the frozen scorer has successfully completed, preserve the decision in a new file:

```sh
python scripts/selection_decision_report.py --output reports/selection-language/allocation-result-v1.json
```

An existing output may be reused only if its bytes are identical. Do not use the final output name for a pending snapshot. Hash bindings identify the consumed rule, original identity, decision document, selection, evaluation identity, every per-condition result, summary and reporter source. The report requires all 128 conditions, all 336 original paired comparisons, source aggregate inventories and across-seed summary inventories, including comparisons that cannot override the allocation decision.

The report checks the recorded evaluation policy, frozen numerical-source hashes, selected-model/result links, and pooled point-difference arithmetic. It uses the original conditional intervals without recomputing them. **This is a consumer of the completed scorer's evidence, not an independent rerun of its integrity checks.** It does not reopen raw corpora, deserialize or rehash model checkpoints, revalidate cached evaluation batches, establish the scientific adequacy of the controls, or recompute inference and confidence intervals. Those responsibilities remain with the unchanged scorer and its existing gates. Publish the full study report alongside this narrower decision.

| Output | Meaning and next action |
|---|---|
| `pending` | Complete selection or summary absent. Wait; no allocation verdict. |
| `invalid` | A record is malformed, a binding fails, an inventory is incomplete, or a required estimate is undefined. Preserve the record and use the frozen recovery rules; no biological conclusion. |
| `pass` | Both analyses meet every fixed criterion. At most one separately declared independent-corpus confirmation is permitted. Nothing is launched automatically. |
| `fail` | The completed recorded study does not clear the allocation rule. Stop anatomical subset searches in this program and publish all outcomes. This is not evidence of equivalence or a general rejection of biological wiring. |

The four-cell mean threshold is inclusive at −0.005 BPB, without rounding. Cell means, interval upper endpoints and all twelve individual contrasts must be strictly below zero. Official and overlap-filtered analyses must each pass; they are not independent replications. Output lists every required cell, individual difference and boolean criterion, so a small mean improvement cannot hide an uncertain cell or unfavorable rewire.

Exit code 2 means an invalid record; code 0 means the reporting operation succeeded, including pending and valid fail. Automation must inspect the JSON `decision`, not interpret a zero exit code as scientific success. Completion of this decision does not complete the paused overall FLM goal.

Five [small synthetic checks](selection-decision-report-test.py) cover the exact margin, an interval touching zero in the filtered analysis, an unfavorable individual rewire hidden by a favorable family mean, undefined/nonfinite/missing/duplicate contrasts, and a complete 128-record consumer fixture with a damaged result. They import no model or corpus code:

```sh
python -m unittest discover -s tests -p test_selection_decision_report.py -v
```

These checks passed in 3.6 seconds on 12 September. The live repository invocation returned pending. This validates reporting behavior; it adds no language, biological or training result.

## Mac execution amendment

The user subsequently requested execution on the MacBook. The [execution amendment](selection-mac-execution.md) retains the original design and allocation thresholds but uses all 128 fresh native Mac fits, with a separate identity and disclosed initialization changes. The original Windows attempt is retained; do not combine its fits with the Mac cohort.

Use the explicit Mac mode for the active study:

```sh
python scripts/selection_decision_report.py --mac-study
# Only after the full native cohort and scorer complete:
python scripts/selection_decision_report.py --mac-study --output reports/selection-language/mac-v1/allocation-result-v1.json
```

This mode verifies the original identity, the native identity's permitted differences, the amendment document, runtime profile, initialization probe and execution adapter. It then uses the same gate calculation on the separate Mac results. Graphs, seeds, allocations, budgets, data, comparisons and thresholds cannot change through this mode. A sixth synthetic check rejects changed budgets or graph identities in the Mac lineage. Default mode continues to report the historical Windows attempt and must not be used as the active study monitor.
