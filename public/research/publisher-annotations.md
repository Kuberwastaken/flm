# Reconcile circuit metadata with the publisher

The [selection feasibility study](selection-study.md) needs more than type-name
prefixes to identify a functional circuit. We have now acquired the publisher's
curated body annotations and its separate neurotransmitter table, pinned their
object generations, and joined both to the frozen runtime by body ID. This
establishes annotation provenance and exposes missing information. It does not
establish a complete circuit or change a trained graph.

## Sources and identity checks

The [MaleCNS download documentation](https://male-cns.janelia.org/download/)
distinguishes curated neuron annotations from transmitter predictions. Its two
body-level Feather files total **57,766,148 bytes**. Both downloaded files matched
their server-provided MD5, and the [acquisition card](publisher-annotations/source-card.json)
records their SHA-256, exact object generation and generation-pinned HTTPS URL.
Raw Feather files remain in the ignored local acquisition cache under CC BY 4.0.

The curated table contains 211,577 rows and the transmitter table 1,835,518 rows.
These are table/segment counts, not comparable estimates of complete neurons in
the animal. Both contain entries outside the 166,700-neuron acquired runtime.
The reconciliation checks unique publisher body IDs and verifies the runtime's
metadata, body-ID and sign arrays against the frozen graph-card hashes.

## Findings

| Check | Observed result |
|---|---:|
| Runtime IDs found in curated annotations | 166,700 / 166,700 |
| Exact type-label matches | 166,700 / 166,700 |
| Exact superclass matches | 166,700 / 166,700 |
| Runtime side matches soma side alone | 148,833 / 166,700 |
| Runtime side matches soma side, otherwise root side, otherwise empty | 166,700 / 166,700 |
| Runtime IDs found in transmitter table | 166,522 / 166,700 |
| Runtime label matches publisher consensus where a record exists | 166,522 / 166,522 |
| Runtime label matches per-body prediction where a record exists | 146,396 / 166,522 |
| Nonempty publisher `ground_truth` fields | 85,484 / 166,522 |
| Runtime label matches those nonempty `ground_truth` fields | 85,484 / 85,484 |

The 17,867 side differences are explained by the soma/root fallback; they are not
unexplained opposite-hemisphere assignments. Root side and soma side still have
different meanings and remain separate columns in the released records.

For the 178 bodies without a transmitter record, the runtime stores the literal
string `nan` and assigns fast sign zero. Those rows are explicitly listed rather
than counted as successful transmitter matches. The current signed-edge rule
also agrees with the publisher consensus for all 166,522 present records: +1 for
acetylcholine, −1 for GABA or glutamate, zero otherwise. This verifies an existing
modeling rule, not physiological correctness of a universal excitatory/inhibitory
sign mapping.

Per-body predictions disagree with runtime/consensus for 20,126 present records.
The runtime is consistent with consensus, not a simple per-neuron argmax. The
publisher's `ground_truth` field is source-provided evidence, not an independent
experimental confirmation by this project. Prediction, consensus, curated class
and that field are all retained separately.

## Consequences for the next selector

All 316 PAM and 16 PPL1 candidates have curated class `DAN` and a nonempty
publisher `ground_truth` field of dopamine. They remain zero-sign under the
current fast recurrence, so selecting them alone would not restore a modulatory
learning mechanism. The two DPM-labeled candidates also have a dopamine consensus
in this source, but **neither has a nonempty `ground_truth` field**. We retain that
provenance distinction and do not infer established DPM physiology from it.

The publisher identifies **686 ALPN-class neurons** in the same `cb_intrinsic`
population. These were absent from the previous named mushroom-body inventory.
They are now included in the candidate annotation export as potential upstream
partners, without assuming that every ALPN innervates the chosen circuit or
adding them automatically to a functional selector.

There is also useful instance-level information that the runtime table omitted.
For example, publisher strings include `MBON14(a3)_R` and
`MBON11(y1pedc>a/B)_L`. These are retained verbatim. Their textual notation is not
a synapse-level compartment map; interpreting it requires the publisher's cell
typing and the relevant anatomical literature. The body tables do not contain
the full per-synapse compartment connectivity needed to certify preservation of
a chosen pathway.

Next, use these verified IDs and labels to define explicit circuit roles and
verify their connections/compartments. Declare inclusion, external-input and
modulatory rules before language fitting. Keep the functional selector and its
size-matched ranking/random/rewiring controls in a new study identity. The
existing BabyLM, topology and sensory studies remain unchanged.

## Reproduce and inspect

```powershell
python -m unittest discover -s tests -p test_publisher_annotations.py -v
python -m flm.publisher_annotations --acquire
```

The acquisition command reuses exact cached files or fetches the pinned object
generation and verifies its SHA-256. It does not refresh pins or replace the
runtime data. It needs the existing language extra (`pyarrow==19.0.1`) and the
previously acquired runtime metadata. Three fixtures check ID-based joins,
missingness, duplicate rejection, side precedence, distinct transmitter fields
and the fast-sign mapping.

The [summary](publisher-annotations/summary.json) includes complete
coverage counts and per-family provenance. The
[5,293 candidate rows](publisher-annotations/candidate-annotations.csv)
cover the prior named families and the ALPN class. The
[178 missing transmitter records](publisher-annotations/missing-transmitter-rows.csv)
retain their runtime identities and labels. These are derived audit records;
no text corpus, training windows or held-out language scores are read.
