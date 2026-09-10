# Structural audit of the language controls

These are the four frozen 1,024-neuron, 76,130-edge graphs used by the ongoing
WikiText topology study. The audit reads their arrays and provenance, without
reading model losses or changing the graphs, training protocol or selections.
The smaller 256-neuron cue/context graphs are a separate experiment.

| Graph | Measured edges retained | Reciprocal directed edges | Strong components | Largest component |
|---|---:|---:|---:|---:|
| Measured | 100.00% | 54.19% | 86 | 939 |
| Rewired 101 | 12.80% | 9.23% | 86 | 939 |
| Rewired 103 | 12.88% | 9.32% | 86 | 939 |
| Rewired 107 | 12.84% | 9.40% | 86 | 939 |

Every graph retains the three original self-edges. Reciprocity counts non-self
directed edges with an existing reverse edge, divided by all non-self directed
edges. Strong components are maximal sets with directed paths between every
pair of nodes. Component equality is observed here, rather than imposed by the
swap rule. All graphs have 85 nodes with no outgoing recurrent edges, zero nodes
with no incoming edges, and no isolated nodes. Those 85 nodes still receive
input and contribute pooled features; the count concerns this selected subset,
not the neurons' connections in the complete anatomical dataset.

All three controls reproduce exact per-node incoming/outgoing degrees, signed
incoming counts, incoming weight/contact multisets, source signs, original self
edges, identities, positions and pool membership. The maximum per-node absolute
change in each of the degree/count/incoming-strength checks is zero. The CSV
contains all 4,096 graph/node records, including assumed source sign and body ID.

Outgoing weighted strength is allowed to change, and changes at all 939 nodes
with outgoing edges. Mean absolute changes across all 1,024 nodes are 0.22073,
0.21871 and 0.21675 for seeds 101, 103 and 107; maxima are 1.36696, 1.38267 and
1.74914. These use the graph's stored base magnitudes, before learned gains;
they are not synapse counts or biological conductances. Holding incoming
weight multisets fixed does not hold outgoing strength fixed.

Each null completes 761,300 accepted swaps. Proposal counts are 1,967,018,
1,970,044 and 1,966,758. Pairwise overlaps between null graphs are 12.577%,
12.554% and 12.446%. No graph was selected using these diagnostics or language
scores. Distinct seeds and low edge overlap do not establish uniform null
sampling or adequate mixing of the constrained swap chains.

The published matrices use the same original node order and show positive and
negative edge presence with a common categorical palette. They show neither
weight magnitude nor inference activity. Every matrix has 1,024 rows and 1,024
columns; the downloadable 3,200-pixel figure preserves detail lost when the
browser scales it down. The indexed NPZ arrays are the authoritative graph data.

This comparison asks about higher wiring organization conditional on preserved
anatomical features. It does not compare all anatomy against no anatomy. These
structural differences establish that the control graphs changed as specified;
they do not establish a language-prediction benefit. That requires the completed
matched training and test comparison in the declared protocol.

## Reproduction and downloads

From the FLM source checkout:

```powershell
python -m flm.language_structure
python scripts/language_structure_figures.py
python -m unittest discover -s tests -p test_language_structure.py
```

The audit rejects changes to frozen graph bytes, null cards, numerical sources
or protocol. It recomputes the graph-card topology statistics and invariant
checks, emits the complete per-node CSV, and verifies the public ZIP's CRC and
every archived file against its SHA-256 manifest. It does not read the language
corpora or checkpoints. The figure generator compares its independently
recomputed report against the saved audit before plotting all four graphs.

- `public/research/language-topology-graphs.zip`: four exact training graph NPZs,
  their cards, the protocol and identity, structural measurements, node CSV,
  audit source, this note and source/license notices. Corpus text and trained
  weights are excluded. Full training reproduction also needs separately
  acquired data and the source checkout.
- `public/research/language-topology-structure.json`: graph hashes, reproduced
  invariants, component sizes, outgoing-strength changes and every overlap.
- `public/research/figures/language-topology-nodes.csv`: all node statistics.
- `public/research/figures/language-topology-matrices.png` and `.svg`: generated
  signed matrices, with unchanged node order.

The bundle retains CC BY 4.0 attribution for graph data and labels rewired
contacts as artificial transferred magnitudes. FLM's original audit code is MIT.
