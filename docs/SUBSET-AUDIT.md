# Why these 1,024 neurons?

The language graph is a computationally bounded, connectivity-ranked subset.
It is **not an intact functional circuit or a miniature whole brain**. Its
selection predates the recorded language experiments. The large amount of cut
connectivity limits what a language topology result can say about biology.

The [machine-readable audit](../reports/subset-audit/summary.json),
[per-neuron table](../reports/subset-audit/nodes.csv) and
[cell-type coverage](../reports/subset-audit/cell-types.csv) make the selection
and boundary losses inspectable. All counts below concern the acquired MaleCNS
graph, not every connection in a biological specimen.

## Selection and recorded history

The unchanged [selector](../flm/graph.py) applies these rules:

1. Start with 166,700 acquired neurons. Only the 32,164 source-labeled
   `cb_intrinsic` neurons are eligible.
2. Rank eligible neurons by incoming **plus** outgoing raw synaptic contacts
   to other eligible neurons. A self connection contributes to both terms.
   Body ID breaks ties. Language data and scores are not selector inputs.
3. Take the first 1,024: 0.6143% of acquired neurons, or 3.1837% of eligible
   neurons. The retained/excluded strength boundary is 7,862/7,858 contacts.
   The size was a CPU/browser design choice; no functional completeness
   criterion or performance-based search over subsets is recorded.
4. Order selected neurons by cell-type label, then body ID. Retain only
   connections whose two endpoints are selected.
5. Exclude outgoing connections from sources assigned fast sign zero.
   Initialize remaining weights with `log1p(contact_count)`, normalized by
   each target's absolute incoming sum, then apply the presynaptic sign.
6. Form 128 consecutive pools of eight neurons for the readout. These are
   computational pools, not measured biological compartments.

The graph's only recorded file commit is `7b78e2f`, at 02:56:20 +05:30 on
10 September 2026. The first trainer commit, `e6b93e6`, follows at 03:08:12.
This supports the recorded ordering and absence of subsequent graph changes;
Git history cannot rule out unrecorded design decisions.

The source sign convention is acetylcholine +1, GABA/glutamate −1, and
other/unclear 0. The subset has 490 positive, 449 negative and 85 zero-sign
neurons. Zero-sign neurons still receive learned text input, maintain state
and contribute to pooling; their outgoing anatomical connections are omitted.
This sign abstraction is a modeling assumption, not a full transmitter model.

## How much connectivity is cut?

Here a *connection* is a directed neuron pair and *contacts* are its raw
synaptic count. The first three rows partition all connections touching the
selected set; an internal connection is counted once in that union.

| Relation | Directed connections | Raw contacts |
|---|---:|---:|
| Between selected neurons, before sign filtering | 88,457 | 1,967,664 |
| Excluded source → selected target | 847,059 | 7,595,687 |
| Selected source → excluded target | 799,007 | 5,588,715 |
| Internal connections removed by the zero-sign rule | 12,327 | 206,669 |
| Internal connections used by the recurrent model | 76,130 | 1,760,995 |

Boundary selection removes **79.42% of incoming contacts**:
7,595,687 / (7,595,687 + 1,967,664). It removes **73.96% of outgoing contacts**:
5,588,715 / (5,588,715 + 1,967,664). These have different denominators and
exclude the additional internal sign-filter loss. The recurrent model retains
11.62% of the 15,152,066 contacts touching the selected set, counting internal
contacts once.

These are contact-count fractions, not estimates of functional current,
information or preserved computational capacity. Normalizing remaining weights
does not reconstruct the removed inputs or feedback. Learned text projections
also do not replace the missing neurons physiologically.

## Which circuits survive?

There are 506 nonempty cell-type labels, plus an empty label, among the selected
neurons. The following are literal prefix counts in the source metadata; they
are examples rather than certified or exhaustive circuit memberships.

| Source-label prefix | Eligible neurons | Selected neurons |
|---|---:|---:|
| KC | 4,064 | 0 |
| MBON | 97 | 48 |
| EPG | 50 | 6 |
| PFN | 456 | 0 |
| PFL | 50 | 0 |
| FB | 602 | 34 |
| ER | 282 | 1 |
| LHCENT | 44 | 21 |

The absence of KC-labeled neurons and partial MBON coverage are particularly
important: the language model cannot be described as an intact mushroom-body
learning circuit. KCs are the intrinsic cells and MBONs an output class of
that circuit ([Schlegel et al., 2024](https://www.nature.com/articles/s41586-024-07686-5));
the incompleteness statement is an inference from that biology and this audit,
not a physiological assay. Partial central-complex labels likewise do not establish a
complete navigation circuit. The audit does not identify an intact circuit
whose biological computation the text model preserves.

## Reproduction and interpretation

With the six acquired source files at the paths and hashes in the unchanged
[graph card](../data/graphs/central-1024/graph-card.json), run:

```sh
python -m unittest discover -s tests -p test_subset_audit.py -v
python -m flm.subset_audit
```

The [auditor](../flm/subset_audit.py) verifies source and graph hashes, replays
the full-population ranking, and exactly matches selected IDs, retained edges,
contacts and initial weights. It streams CSR rows without constructing a dense
full graph. Four tests independently check a hand-counted directed fixture,
self connections, zero-sign losses, empty rows, chunk boundaries and malformed
inputs. The report records the auditor hash and both CSV hashes.

The ongoing [language topology comparison](LANGUAGE-TOPOLOGY-PROTOCOL.md)
tests whether the retained measured topology helps under this particular
selection and model. A favorable result would not validate the excluded
circuitry or the sign abstraction. An unfavorable result would not settle the
utility of other subsets or complete biological circuits. A future subset
comparison needs declared selection rules and budgets before performance is
examined; changing the current graph would invalidate the present comparison.
