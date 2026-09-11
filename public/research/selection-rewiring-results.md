# Complete structural controls for the selection study

All **192 controls completed successfully, with zero failures**: three graph
seeds for each of the 64 original selections. The
[25.4 MiB archive](selection-rewiring.zip) contains those
controls, all 64 originals and the complete manifests. All 256 graphs are
untrained. This finishes structural-control preparation; it does not finish the
selection-language experiment or establish a biological advantage.

The archive is 26,622,579 bytes. Its SHA-256 is:

```text
1b849f068ab1c0bd7e8331e2aa7ddb573ed6173f79e739727b86ac7d176aebd3
```

The [release record](selection-rewiring/release.json) and
[standalone audit](selection-rewiring/standalone-audit.json) bind the
exact archive bytes. The independent NumPy/SciPy auditor was copied outside
the repository and successfully checked the archive there. It verifies all
original/control payload hashes, fixed arrays, directed degrees, source signs,
incoming magnitudes, self edges, unique pairs and the reported structural
diagnostics. It does not replay the swap trajectories. See the
[audit specification](selection-rewiring-audit.md).

## What changed structurally

![Edge overlap and largest strongly connected components across all original selections and their three rewires.](figures/selection-rewiring.png)

Each point represents one original selection, with the mean and full range over
its three graph seeds. Bars are descriptive seed ranges, not confidence intervals.
The 64 selections share source neurons, selection rules and random seeds; they
are not independent biological replications. The
[CSV](figures/selection-rewiring.csv) keeps all 192 cases and
the [summary](selection-rewiring/summary.json) keeps every selection.

| Selection rule | Original selections | Successful controls | Original edges retained, range over all controls |
|---|---:|---:|---:|
| Operational KC-centered | 8 | 24 | 28.96–62.87% |
| Contact-ranked | 8 | 24 | 10.74–14.45% |
| Uniform random | 24 | 72 | 0.81–4.72% |
| Superclass/side/sign-stratified random | 24 | 72 | 3.01–11.70% |

Overlap counts all original edges, including fixed self edges. The same
ten-accepted-swaps-per-edge target therefore leaves very different overlaps in
different selections. Their degree/sign constraints and densities differ too;
raw overlap is not a calibrated measure of randomization across those sets.
Higher overlap alone neither proves poor mixing nor demonstrates preserved
functional computation. Uniform sampling and sufficient mixing remain unproven.

The right panel reports the fraction of selected neurons in the largest
strongly connected component before and after rewiring. The diagonal marks equal
fractions. This statistic can change despite identical directed degrees; equal
component sizes also do not imply identical circuits. Neither panel measures
neural function, language learning or animal behavior.

## What is held fixed

For each original selection, seeds 101, 103 and 107 use the same node identities,
ordering and pools. Swaps preserve directed degrees, incoming sign counts,
incoming signed-weight multisets and original self edges. They target ten
accepted swaps per edge within a ceiling of one hundred proposals per edge.
Every case reached that target. Outgoing weighted strength and higher motifs are
not held fixed. Transferred contacts and weights describe original incoming-slot
magnitudes, not measured synapses between the artificial endpoint pairs.

The [selection rules and boundary audit](circuit-selection.md) remain unchanged.
The high external-contact losses, absent fast modulatory pathways and lack of
validated circuit completeness remain limitations of these models. Generating
rewires does not repair them.

## Reproduce and inspect

The archive contains `source-manifest.json`, `rewiring-manifest.json`,
`original/graphs/<selection>.npz` and `rewired/<selection>/null<seed>.npz`.
No tokenizer or trained weights are included. The source manifest retains
MaleCNS attribution, the pinned runtime revision and the CC BY 4.0 license.

```powershell
python scripts/audit_selection_rewiring.py public/research/selection-rewiring.zip --sha256 1b849f068ab1c0bd7e8331e2aa7ddb573ed6173f79e739727b86ac7d176aebd3
python -m unittest discover -s tests -p test_selection_rewiring_report.py -v
python -m scripts.selection_rewiring_report
```

The [report generator](selection-rewiring/selection_rewiring_report.py) binds the archived
manifest's exact checksum and requires the local parsed manifest to agree. It
allows native newline differences, not changed records. Four tests cover the
aggregation, seed ranges, component denominators, retained failures, incomplete
inventories and manifest identity checks. The figure's
[provenance record](selection-rewiring/figures.json) binds the CSV,
PNG and SVG bytes. All plotted values come from the audited receipts.

The next step is the [training-cost pilot](selection-timing-pilot.md), followed
by a frozen matrix of selections, rewires and training seeds. Selector contrasts
and within-subset wiring contrasts remain separate. No selector may be chosen
using held-out language outcomes. BabyLM remains the active priority training
queue; the previous language and behavior results are unchanged.
