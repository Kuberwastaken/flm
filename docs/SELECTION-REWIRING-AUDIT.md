# Independent checks for selection-study rewiring

The [64 exported selections](SELECTION-GRAPH-EXPORTS.md) are the inputs to three
rewiring controls each, using seeds 101, 103 and 107. These **192 controls are
untrained graphs**, not 192 language fits. The final language matrix and budget
remain undecided. All previous language and behavior studies remain unchanged.

The dated [preflight record](../reports/selection-rewiring/preflight.json) checks
**101 completed controls, with zero recorded failures**, from the queue snapshot
at 09:38:25 UTC on 11 September 2026. This is an immutable partial snapshot, not
a live completion counter. The final archive gate correctly rejects it.

## What is checked independently

[audit_selection_rewiring.py](../scripts/audit_selection_rewiring.py) uses NumPy
and SciPy, without importing FLM, Torch or the swap generator. For every supplied
original/control pair it checks:

- Exact array inventory, shapes, dtypes and fixed values, including neuron
  identities, positions, pool assignments, incoming-slot contacts and weights.
- Directed incoming and outgoing degrees, source-sign constraints, incoming
  signed-weight multisets and fixed self edges.
- Valid endpoints, unique body pairs, finite weights and changed topology.
- Independently recomputed strongly connected components, reciprocity, original
  edge overlap and unchanged endpoint slots against the generator's receipt.
- Source-graph checksums, graph seed and declared swap-budget arithmetic.

The generator targets ten accepted swaps per edge with at most one hundred
proposals per edge. The audit checks those reported counts and their arithmetic;
it **does not replay the random swap trajectory**. Neither reaching that target
nor observing changed topology establishes uniform sampling or adequate mixing.
Outgoing weighted strength and higher-order motifs are not held fixed.

Rewired contacts are transferred incoming-slot magnitudes. They are not new
measurements of synapses between the artificial endpoint pairs. This convention
holds input magnitudes fixed while testing a particular wiring null model.

## Publishing a complete inventory

[package_selection_rewiring.py](../scripts/package_selection_rewiring.py) refuses
to publish a final bundle until every original graph and seed has a terminal
record. Explicit failures remain in the manifest; they are never silently
omitted or counted as successful controls. An OS lock excludes an active
generation/resume writer while the final package is built.

The packager verifies the released source manifest, generation source hashes,
per-case receipts and graph bytes, then builds a temporary archive. The
independent auditor verifies the full archive before it replaces the final
path. It rejects missing, extra or duplicate entries, incorrect completion
totals, altered diagnostics and altered graph payloads. Failed cases are
reported, not independently rerun. A failed check preserves an existing final
archive.

The eventual bundle contains all original graphs, every successful rewire and
all terminal receipts. It contains no tokenizer, trained weights or language
scores. A terminal preparation record may include failures; it is not a claim
that all controls succeeded or that a training protocol has been frozen.

## Reproduction

From the repository, run the fixture checks and a new dated preflight:

```powershell
python -m unittest discover -s tests -p test_selection_rewiring_archive.py -v
python -m scripts.preflight_selection_rewiring --output reports/selection-rewiring/another-dated-preflight.json
```

The preflight reads an atomic progress snapshot and checks its committed cases
without resuming or modifying generation. It refuses to overwrite its output.
Seven tests cover known signed cycles, damaged fixed arrays, degree/sign/self
constraints, duplicates, altered archive diagnostics, checksum failures,
incomplete inventories, retained synthetic failures and end-to-end packaging.
The packaging fixture also verifies that corruption leaves a prior release
untouched. Synthetic failures in tests are distinct from actual queue failures.

After generation exits with a complete terminal inventory:

```powershell
python -m scripts.package_selection_rewiring
python scripts/audit_selection_rewiring.py public/research/selection-rewiring.zip
```

The standalone auditor also accepts `--sha256` with the value from the final
release record. It can be copied outside the repository and run with NumPy and
SciPy against the archive. The final archive is **not available at this partial
checkpoint**.

## What this enables, and what it cannot establish

Selector comparisons ask whether the operational KC-centered rule helps under
the same language-model interface relative to connectivity-ranked and matched
random selections. Within-subset rewires ask whether retained wiring helps
beyond the null model's preserved properties. They are separate contrasts:
matching neuron count or annotation strata does not match density or parameter
allocation across subsets.

Training-only cost profiling must precede the final fit matrix. That matrix must
declare selections, graph seeds, training seeds, common exposure, validation
selection, all planned contrasts and the whole-study test gate before fitting.
The anatomical membership rule must not be chosen using language outcomes.
These controls do not repair missing external inputs or supply the biological
plasticity mechanisms absent from the current fast/slow rate model.
