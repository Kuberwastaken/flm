# Untrained graph exports for the selection study

The subsequent [complete structural release](selection-rewiring-results.md)
contains all 192 successful rewires and all 64 originals. The complete archive
passes the independent audit. No selection language fits have been run.

The [training-cost pilot](selection-timing-pilot.md) is also prepared. All 64
full-size graph configurations pass tiny synthetic backward/update checks;
official corpus timing waits for the priority queues to finish.

The [KC-centered candidates and comparison selections](circuit-selection.md)
are now exported into the existing FLM graph format. The
[5.6 MiB graph archive](selection-graphs.zip) contains all
64 graphs and their manifest. It contains **no trained weights or tokenizer**.
These are model-ready anatomical subsets, not usable language checkpoints or
certified intact biological circuits.

## What is held consistent

Every export orders cells by type and then source body ID, applies the same
128-pool contiguous index rule, retains all induced positive body-pair contacts,
and removes outgoing edges whose source has fast sign zero. Weights use the
existing float32 `log1p(contact_count)` rule followed by absolute incoming
normalization and source sign. The membership threshold does not filter retained
edge contacts once cells have been selected.

The interface check instantiates the actual FLM implementation with a
4,096-entry vocabulary, 96-dimensional embeddings, 128 pools, tied readout,
fast/slow state and an input projection to every selected node. No candidate
receives a role-specific input or output privilege. This fixes the interface
configuration for preparation; the official language-fit matrix and training
budget are still unfrozen. Consistent pooling code does not make pool composition
identical between different node sets.

The exporter exactly reproduces **all nine arrays and their dtypes** in the
frozen 1,024-neuron graph: row, column, weight, contacts, source sign, body IDs,
source indices, positions and pool indices. Each new graph's node count, fast
edge count and retained contacts agree with the preceding independent inventory
audit. Every saved graph is reloaded and checked through actual FLM forward
computation on three synthetic token IDs. That checks the interface, not
language quality, training convergence or throughput.

## Parameter count is not effective capacity

| Five-contact candidate | Neurons | Fast edges | Allocated trainable parameters | Fast + slow state, batch 1 float32 |
|---|---:|---:|---:|---:|
| KCg-d / L | 487 | 15,674 | 486,384 | 3,896 bytes |
| KCg-d / R | 540 | 18,499 | 494,456 | 4,320 bytes |
| KCg-m / L | 1,202 | 195,007 | 736,502 | 9,616 bytes |
| KCg-m / R | 1,150 | 184,616 | 720,963 | 9,200 bytes |

With this interface the allocation is `422497 + 99*N + E`, where `N` is the
number of nodes and `E` the retained fast edges. Counts come from instantiated
models and include every parameter group in the
[manifest](selection-graph-manifest.json). Across all 64 graphs,
the total ranges from 470,945 to 766,693 entries. This is not a parameter-matched
architecture comparison. State bytes exclude weights, temporary activations,
optimizer state, Python bookkeeping and any input/output buffer.

Sparse random selections make the effective-capacity issue especially visible.
At the 487-node budget, uniform seed 203 has only 235 fast edges, **331 nodes
without a fast incoming edge** and **109 with exactly one**. The corresponding
KC-centered candidate has 15,674 edges, three nodes without incoming edges and
23 with exactly one. Nodes without recurrent input can still receive learned
token input and retain temporal state; they are not necessarily inactive.

For a row with one incoming edge, incoming normalization fixes its weight to the
source sign. Its sole edge-gain entry cannot change that relative magnitude.
A test checks this invariance in the actual model while confirming that
multi-input rows can change. Counting allocated edge-gain entries as equally
useful degrees of freedom would therefore be misleading. The manifest records
zero-sign nodes, zero incoming/outgoing degree and single incoming degree for
every selection.

## Rewiring preparation is complete

The [resumable runner](selection_rewiring.py) uses the existing tested
directed-swap implementation to prepare seeds 101, 103 and 107 for each graph:
192 untrained controls, all completed. Each case targets ten accepted swaps per edge,
with a limit of 100 proposals per edge. It preserves directed degrees, source
sign constraints, incoming signed weight magnitudes, original self edges,
node identities, positions and pools. Transferred contacts/weights are input-slot
magnitudes, not measurements of the newly connected body pair.

Each completed payload has a receipt binding its source graph, generator code,
runner code, seed and swap budgets. Resumption validates the receipt, graph hash
and structural invariants before reuse. An interrupted payload without a receipt
is recreated; an infeasible generation is recorded as a failure and is not
silently retried or omitted. The directory has an OS-held writer lock. The
generator has exited; BabyLM training continues. Neither operation changes the
other's data, source graph or training protocol.

The complete 192-control inventory and its structural diagnostics are now
[audited and available](selection-rewiring-results.md), with no failures. Finite
swap chains do not establish uniform sampling or adequate mixing. The actual
fit matrix remains undecided. Cost measurements must use a controlled compute
window; the synthetic export checks provide no performance benchmark.

A [dated initial receipt audit](selection-rewiring-started.json)
independently checks the first 38 completed controls using NumPy: degrees,
source-sign compatibility, duplicate rejection, self-edge preservation, fixed
arrays, changed topology and saved-file identities. It is a historical snapshot,
not a live progress counter or a final inventory audit.

## Reproduce or inspect

```powershell
python -m unittest discover -s tests -p test_selection_graphs.py -v
python -m flm.selection_graphs
python -m unittest discover -s tests -p test_selection_rewiring.py -v
python -u -m flm.selection_rewiring
```

Exported working files live in `work/selection-graphs-v1`; rewiring payloads and
their progress report live in `work/selection-rewiring-v1`. These are generated
intermediates. The published ZIP contains the complete original 64 graph exports
and manifest; the large acquired runtime is not bundled. Reproduction requires
the existing source data, selector inventories and FLM dependencies.

The [export record](selection-graph-release.json) binds the
archive and manifest checksums. Four export tests cover identity/order,
direction/sign filtering, normalized weights, real forward/RNG behavior,
single-input invariance and invalid input rejection. Two rewiring-runner tests
exercise real swaps, verified reuse, changed identities, altered payloads,
orphan recovery and retained failure records. Existing numerical training
sources remain unchanged.

The archive preserves MaleCNS attribution and the pinned runtime-derivative
source URL/revision. Anatomical data retain CC BY 4.0; FLM original code is MIT,
by Kuber Mehta. A graph download does not imply demonstrated learning,
biological fidelity or an advantage over conventional recurrent models.
