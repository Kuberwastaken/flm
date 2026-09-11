# Operational KC-centered selections and their controls

The [pathway audit](selection-pathways.md) established that naming cell families
is insufficient to choose a useful subset. This extension implements a concrete
selection rule, records every selected body ID and measures its boundaries. It
also constructs connectivity-ranked and random comparisons at each candidate's
actual size. **These are untrained anatomical inventories, not language results
or certified functional circuits.**

## A reproducible selection hypothesis

Ganguly et al. distinguish visual-associated KCγ-d and olfactory-associated
KCγ-m populations, describe sparse input pathways and use a five-synapse
body-pair threshold in their visual-input analysis. This motivates the seed
populations and one reported threshold here. It does not validate transfer of
their FlyWire/hemibrain circuit membership to MaleCNS, or make our added output
and auxiliary rules their protocol. [Ganguly et al., 2024](https://www.nature.com/articles/s41467-024-49616-z)

Rule `kc-centered-one-step-v1` is implemented in
[circuit_selection.py](circuit-selection/circuit_selection.py):

1. Include **every** cell with literal type `KCg-d` or `KCg-m` on the chosen
   source side, L or R. Do not rank, cap or truncate this seed population.
2. Add every non-KC, non-MBON, non-auxiliary cell with a nonzero fast sign and
   at least one qualifying directed connection onto a seed KC. This is an input
   *partner*, not a claim that its modality or calyx compartment is established.
3. Add every MBON-prefix cell receiving a qualifying connection from a seed KC.
   MBON feedback onto a seed alone does not qualify it as an output.
4. Add every PAM/PPL1/APL/DPM-prefix cell connected in either direction to a seed
   KC or selected MBON, using any acquired positive contact count. This is a
   deliberately permissive auxiliary rule; DPM physiology remains unresolved in
   the acquired annotations.
5. Stop. Do not recursively expand partners, add other KC types, apply a partner
   side filter or impose a size cap. Cross-side connections remain explicit.

The qualifying threshold is separately **1 or 5 contacts on a body pair**.
Eight candidates cover both types, both sides and both thresholds. Threshold 1
is retained as a sensitivity inventory; neither threshold was selected using
language scores. Membership thresholds do not filter the eventual induced
edges: all acquired positive contacts between included cells remain in the raw
graph. Fast-edge counts then apply the existing zero-source-sign exclusion.

## The candidate boundaries are still substantial

| Seed type / side | Membership threshold | Total cells | Seed KCs | Fast edges | Incoming raw contacts cut | Outgoing raw contacts cut |
|---|---:|---:|---:|---:|---:|---:|
| KCg-d / L | 1 | 784 | 99 | 27,049 | 87.64% | 84.83% |
| KCg-d / L | 5 | 487 | 99 | 15,674 | 89.52% | 85.71% |
| KCg-d / R | 1 | 858 | 107 | 33,652 | 87.29% | 84.60% |
| KCg-d / R | 5 | 540 | 107 | 18,499 | 88.87% | 85.04% |
| KCg-m / L | 1 | 1,385 | 689 | 207,081 | 69.65% | 66.49% |
| KCg-m / L | 5 | 1,202 | 689 | 195,007 | 64.85% | 62.05% |
| KCg-m / R | 1 | 1,318 | 653 | 195,764 | 69.18% | 65.83% |
| KCg-m / R | 5 | 1,150 | 653 | 184,616 | 65.63% | 63.70% |

The difference between seed coverage and whole-subset coverage is central:

| Five-contact candidate | Seed KC incoming cut | Seed KC outgoing cut | Input-partner incoming cut | MBON incoming cut | Auxiliary incoming cut |
|---|---:|---:|---:|---:|---:|
| KCg-d / L | 19.23% | 18.49% | 97.38% | 85.62% | 93.70% |
| KCg-d / R | 15.50% | 14.85% | 96.32% | 84.26% | 93.12% |
| KCg-m / L | 3.74% | 3.81% | 96.17% | 58.04% | 80.65% |
| KCg-m / R | 3.58% | 3.83% | 95.80% | 52.76% | 80.89% |

Including connected partners preserves many seed-KC contacts while leaving those
partners heavily dependent on excluded cells. The visual five-contact candidates
contain 301 and 326 auxiliary cells, respectively—more than half their selected
nodes. Shared output and modulatory connections prevent a simple one-hop rule
from isolating a small closed unit. The olfactory candidates preserve more seed
contacts but exceed the old 1,024-neuron budget. Neither result is a reason to
silently truncate a population or call it intact.

These fractions describe **raw contacts**, not functional current or the fraction
of biological computation preserved. Zero-sign auxiliary outputs are still
excluded by the existing fast model. Input partners would need an explicit
external drive in any modeled use; learned language projections do not recreate
their missing biological inputs.

## Matching selections without hiding the population change

The candidates include partners outside `cb_intrinsic`. The control population
therefore uses the fixed union of all six source superclasses found across the
eight inventories: `ascending_neuron`, `cb_intrinsic`, `descending_neuron`,
`ol_intrinsic`, `visual_centrifugal` and `visual_projection`. This contains
**134,491 cells**. The same population is used for every candidate and control.
It is broader than the historic 32,164-cell central-only ranking; the historic
1,024-neuron result is not a matched reference for this comparison.

For each candidate, [selection_controls.py](selection-controls/selection_controls.py) prepares:

- One ranking by exact integer incoming-plus-outgoing raw contacts **within the
  common eligible population**, with self contacts counted twice and body-ID ties.
- Three uniform draws without replacement, matching node count only.
- Three draws without replacement stratified by the joint combination of source
  superclass, source side and fast sign, matching the candidate's exact counts.

Both random procedures use NumPy PCG64 seeds 201, 203 and 207, with populations
sorted by body ID and stratified buckets processed in a fixed order. Candidate
members remain eligible for random sampling. All overlaps are reported. Reusing
seeds across candidates does not create independent biological replications.

There are **64 inventories: eight candidates plus 56 comparison selections**.
This is not a declaration of 64 language fits. The complete
[control summary](selection-controls/summary.json) retains all
conditions and their exact memberships, boundaries and strata. The five-contact
cases illustrate why further controls matter:

| Candidate budget | Candidate fast edges | Ranked fast edges | Uniform fast edges, all three seeds | Stratified fast edges, all three seeds |
|---|---:|---:|---:|---:|
| KCg-d / L, 487 | 15,674 | 15,238 | 235–279 | 784–1,000 |
| KCg-d / R, 540 | 18,499 | 18,431 | 314–355 | 950–1,053 |
| KCg-m / L, 1,202 | 195,007 | 79,309 | 1,480–1,694 | 10,952–11,759 |
| KCg-m / R, 1,150 | 184,616 | 73,563 | 1,496–1,550 | 10,208–10,735 |

Stratum matching still leaves a large density difference. With one trainable
parameter per edge, it also leaves a capacity difference. A future candidate win
over these random subsets could reflect the selection rule, density, composition
or parameter allocation. It would not isolate measured topology. The near edge
counts of the two visual candidates and ranked references also do not establish
matched degrees, weights, signs, function or language performance.

## Decisions before fitting

The operational rule makes candidate membership inspectable; it does not finish
the functional-subcircuit question. Export and rewiring are complete; the
remaining decisions and their structural prerequisites are:

1. Use the [64 exported graphs](selection-graph-exports.md), with their recorded
   source identities, common ordering/pooling and input/readout policy. Actual
   cost profiling remains necessary before choosing complete training groups.
2. Preserve the [192 audited rewires](selection-rewiring-results.md), with their
   specified directed degrees, source signs, incoming magnitudes, self edges
   and pools. Keep selector contrasts separate from topology contrasts.
3. Declare how biological boundary truncation limits the hypothesis. If a more
   compartment-specific selector replaces this rule, retain these inventories and
   state the anatomical reason before any language comparison.
4. Freeze the final subset/rewiring/training-seed matrix, common exposure,
   validation selection and whole-study test gate. Do not choose a subset by
   looking at its test loss or its most attractive continuation.
5. Test role-specific inputs, modulatory dynamics and biological plasticity in
   separately controlled extensions. The current fast model cannot restore
   dopamine-mediated learning merely by including dopamine-labeled neurons.

BabyLM remains the priority training queue. Existing language topology,
computation, sensory and physical studies are unchanged.

The [graph export stage](selection-graph-exports.md) has since completed all 64
model-ready graphs, exact frozen-format compatibility and synthetic forward
checks. The graph archive and full parameter/degree cards are available there.
Three rewires per selection are now [complete and independently audited](selection-rewiring-results.md);
cost profiling and the actual language-fit matrix remain pending. These exports do not certify that
the biological boundaries above have been repaired.

## Reproduction

```powershell
python -m unittest discover -s tests -p test_circuit_selection.py -v
python -m flm.circuit_selection
python -m unittest discover -s tests -p test_selection_controls.py -v
python -m flm.selection_controls
```

Six fixtures check complete seed membership, edge direction, cross-side partners,
auxiliary feedback, no recursive expansion, per-pair thresholds, integer ranking,
exact strata, reproducibility under source reordering and invalid candidates.
Sparse-submatrix control accounting independently reproduces every candidate's
previous full-scan edge/contact counts and boundary fractions. No corpus or
checkpoint is opened.

The [candidate summary](circuit-selection/summary.json),
[body IDs](circuit-selection/candidate-body-ids.json),
[cell roles](circuit-selection/members.csv) and
[all comparator IDs](selection-controls/control-body-ids.json)
bind their inputs and source hashes. These records retain every inspected
candidate, including its inconvenient size, boundary losses and modeling gaps.
