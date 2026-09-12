# Selection-language study v1: bilateral visual-KC candidates

Protocol written 12 September 2026 before any fit in this selection study.
The immutable study identity binds this file and all model, graph, corpus,
validation and evaluation sources. Author: Kuber Mehta.

## Question and scope

Does selection around a declared Kenyon-cell population change language
prediction relative to connectivity-ranked and random selections at the same
neuron count? Within each selected node set, does measured wiring outperform
three degree- and source-sign-constrained rewires?

This is a controlled first screen of operational truncated graph selections.
It is not a test of an intact brain, native mushroom-body learning, visual
behavior or conversational ability. The original 1,024-neuron graph excludes
Kenyon cells; these groups address a different anatomical selection question.

## Frozen choice of complete groups

Register `KCg-d-L-t5` and `KCg-d-R-t5`, respectively 487 and 540 neurons.
They retain all 99 left or 107 right literal KCg-d seed cells plus the partners
defined by the already released five-contact membership rule. Group names
refer to the seed side, not a hemisphere-pure induced graph. Membership and
all graph arrays are taken unchanged from the released catalog.

Visual Kenyon-cell circuitry motivates this choice; the published connectivity
study finds structured upstream inputs together with sparse input sampling.
That supports comparing structure and random controls, not assuming anatomical
superiority. See [Ganguly et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11228034/)
and the local [pathway audit](SELECTION-PATHWAYS.md).

The two candidates preserve about 81–85% of raw contacts incident on their
seed KCs, but cut 89.52% and 88.87% of incoming contacts to the whole selected
node sets. Auxiliary partners account for 301 and 326 cells, respectively.
The fast-sign rule excludes zero-sign modulatory outputs. These limitations
remain even if the candidate wins. See the [anatomical records](../reports/circuit-selection/summary.json).

Both sides are included to avoid selecting a favorable hemisphere. The pair
also has the lowest complete-group cost in the measured catalog. Neither
selection-language training nor validation/test scores were observed when
choosing these groups. Completed BabyLM results concern the old ranked graph;
they do not determine the candidate selection.

Each group includes all eight selectors: the operational candidate, contact
ranking, three uniform selections (201/203/207), and three selections matched
on class, side and fast sign (201/203/207). Each has measured wiring plus all
three rewires (101/103/107), trained at seeds 42 and 43: **128 fits in total**.
No individual selector, rewire or training seed may be dropped. Candidate and
control node counts match within each group; edge and parameter counts need
not match across selectors. Within a selected node set, rewires preserve
allocation, directed degrees and source-sign constraints. The complete
[comparison design](SELECTION-LANGUAGE-COORDINATOR.md) remains applicable.

## Data, updates and validation

Use the pinned BabyLM 2026 `train-10m` cache and its train-only 4,096-token
lexicon. Use no pretrained weights, teacher corpus or inherited pilot weights.
Graph initialization and non-edge initialization follow the released adapter:
learned tensors match within each subset/seed across rewires, and non-edge
tensors match across same-size selectors. All inputs and initialization hashes
must be verified at registration.

Every fit receives exactly 3,000 BPTT/AdamW updates, batch 16, sequence 96,
16 context-warmup positions, four CPU threads and float32 computation.
Score boundary targets during training, as measured by the selection pilot.
Use learning rate 0.002 decaying to 0.0002 by the shared cosine schedule,
100 warmup updates, weight decay 0.01 and gradient norm clip 1.0.
This is 4,608,000 presented and 3,840,000 supervised tokens per fit;
actual byte exposure must match across conditions within each training seed.

Save checkpoints every 500 updates. Only after all 128 fits complete, evaluate
all six checkpoints per fit on the fixed 48-prefix validation panel and choose
the earliest exact minimum BPB. No adaptive stopping, budget extension, seed
replacement or selection of a subset from incomplete results is allowed.
A failure retains its files; a compatible interrupted fit resumes its recorded
optimizer, random state and exposure. Any required method change needs a
separate documented amendment or study, not a silent rewrite of this identity.

Three thousand updates are an explicit limited-exposure screen, not evidence
of convergence or a replacement for the 12,000-update BabyLM comparison.
The training boundary mask also differs. Report this distinction in every
comparison with the old baseline. Longer-exposure confirmation and the other
candidate types/thresholds remain later work; this screen does not complete
the broader FLM objective.

## Held-out analysis

Freeze all 128 checkpoint selections before loading held-out tokens. Apply the
existing [evaluation policy](SELECTION-LANGUAGE-TEST.md) verbatim: full publisher
test partition and separately the fixed overlap-filtered blocks, per-source
and pooled byte-weighted likelihood, state reset at each block and carried
within it, batch eight and 96-token chunks. Exclude boundary targets at test.

Report every individual result and the predefined selection and topology
contrasts. Candidate minus ranked, mean uniform and mean stratified scores
address selection with unequal edge allocation. Measured minus mean rewire
scores address topology within each selector. Retain all individual comparisons,
not just means. Family averages are means of losses, not probability ensembles.
Negative first-minus-second BPB favors the first condition.

Use the source-bound 10,000-draw paired block bootstrap within source components
(seed 31415). These are unadjusted descriptive intervals conditional on the
chosen graphs/checkpoints and artificial blocks with unknown dependence.
Two training seeds and three graph seeds are distinct levels of variation;
hemispheres and rewires must not be counted as independent animal or training
replications. Interpret null, positive and negative outcomes narrowly.

## Cost and execution

The [64-original timing pilot](../reports/selection-pilot/timing.json) completed
before this choice. The [cost decision](../reports/selection-pilot/cost-decision-v1.json)
records all eight group estimates, including the unchosen groups. The selected
128-fit matrix estimates 35.37 update-hours; a 1.5 allowance gives 53.06 hours
against a 72-hour training planning budget. These figures are planning
extrapolations, not measured whole-study runtime. Rewires, seed 43, loading,
validation, I/O and sustained thermal changes were not separately timed.
Held-out evaluation time is additional. Do not truncate slower conditions to
meet a wall-clock deadline.

The exact [request](../reports/selection-language/request-v1.json) is registered
by `python -m flm.selection_language_study initialize --request
reports/selection-language/request-v1.json`. Train the complete inventory with
`python -m flm.selection_language_study train`, then freeze validation choices
with `python -m flm.selection_language_study select`, and score with
`python -m flm.selection_language_test`. Each stage must exit successfully
before the next begins. Preserve sequential commits, all outcomes and source
identities throughout the study.
