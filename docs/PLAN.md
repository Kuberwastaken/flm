# FLM implementation and research plan

Started 10 September 2026; status consolidated 11 September 2026 (Asia/Kolkata).

## Objective and scope

Develop and evaluate FLM as a language model trained from scratch, with reproducible anatomical graph construction, documented data, matched comparators and ChatFLM at [flm.kuber.studio](https://flm.kuber.studio). The working compact release is a starting point for broader data, capacity scaling, alternative learning rules and controlled language-to-action transfer. Completing its artifacts does not complete that broader objective.

The central question is whether measured neural wiring or its associated computation supplies a useful prior for prediction. Report positive, null and negative results against appropriate controls. The present model is a small text completion system; its articulated fly view does not establish an intact simulated brain, conversational competence or language-driven motor skill.

## Current order of work

1. **Finish the fixed BabyLM comparison and its declared evaluation.** Seven of twelve fits are complete: all FLM, GRU and transformer runs at 10M, across seeds 42/43, plus 100M FLM seed 42. The 100M GRU seed-42 run is active. Preserve the registered inventory and common test gate.
2. **Test selection before further dataset or model scaling.** After that queue finishes and its trainer exits, measure selection costs, choose complete comparison groups for anatomical and computational reasons, freeze the protocol and run every registered control. No selection-language result exists yet.
3. **Keep later studies separate.** Alternative language learning, symbolic instruction transfer and food-task adaptation have distinct preparation, budgets and inferential questions. They do not replace the selection comparison. Broader datasets and capacity scaling remain part of the eventual objective.

The [BabyLM validation snapshot](../public/research/babylm-validation.json) is dated evidence, not a live process monitor or held-out comparison. Detailed completion audits remain in [the BabyLM reports](../reports/babylm/); avoid copying changing step counts into this plan.

## Completed evidence to preserve

| Study or artifact | Result and limit |
|---|---|
| Original graph and language architecture | The [subset audit](SUBSET-AUDIT.md) reconstructs 1,024 neurons ranked within 32,164 eligible central neurons. It cuts 79.42% of incoming and 73.96% of outgoing raw contacts and contains no KC-prefix cells. This computational selection is not an intact functional circuit. |
| WikiText baseline | All six fits and test evaluations are complete. Two-seed mean test BPB: FLM **1.9744**, GRU **1.9049**, transformer **1.8767**; lower is better. FLM trails both matched comparators. See [complete scores](../public/research/test-results.json). |
| Topology and slow state | All eight new fits and ten checkpoint selections completed before new-control test scoring. Measured minus rewired averages **+0.001559 BPB**, with all six point estimates positive and three conditional article intervals including zero. Retaining slow state helps against retrained no-slow controls by **0.010999 BPB** on average. See [findings](../reports/language-topology/summary.json). |
| Language computation controls | Six new fits and eight frozen selections are complete. Full-minus-control means are **−0.005797 BPB** for fixed dynamics, **−0.006391** for no lateral recurrence and **−0.124511** for no temporal state. See [findings and parameter limits](LANGUAGE-CORE-RESULTS.md). |
| Recurrent dynamics | The [seven-condition pulse diagnostic](LANGUAGE-DYNAMICS-FINDINGS.md) measures actual initial/trained WikiText cores and acute parameter swaps. Numerical replay establishes neither global stability, language improvement nor behavioral transfer. |
| Sensory and earlier physical studies | The initial five-rule sensory study, [60-run context/wiring extension](WIRING-RESULTS.md), forty choice replays and [27-condition pose-feedback assay plus repeat](CLOSED-LOOP-REPRODUCTION.md) are complete. Results are task-dependent; the trained pose controllers match scripted body paths. These studies use separate sensory networks. |
| Food reference and recorded replay | The [seven-case scripted odor reference](FOOD-APPROACH-RESULTS.md) and [24-case initial/language-trained core reference](FOOD-CORE-PHYSICAL-RESULTS.md) are complete. The latter retains wrong-source contacts and timeouts, with **zero food-task updates** and no demonstrated transfer benefit. All 4,824 recorded observations are audited; all 24 conditions have browser replay with actual saved neural states and reconstructed body geometry. |
| Selection structures | [64 original graphs](SELECTION-GRAPH-EXPORTS.md) and [all 192 rewires](SELECTION-REWIRING-RESULTS.md) are exported and audited, with zero failed rewires. They are untrained structural controls, not language or throughput results. |
| Product and reproduction | ChatFLM, model bundles, five working papers, computational figures, attributed anatomy/body assets and standalone audits are published. The [README](../README.md) links the current releases; [browser verification](BROWSER-VALIDATION.md) and [physical replay release](../reports/food-core-physical/live-release.json) retain dated checks. |

The topology result tests this ranked subset and setup, not biological topology in general. Three graph seeds and two training seeds share two measured references; they are not six independent replications. The exploratory design estimates no topology-by-slow-state interaction. Computation-control benefits likewise do not establish anatomical advantage: conditional article intervals and unequal trainable/gradient-connected parameter counts limit their interpretation. Preserve each study's frozen sources, selections and full outcomes rather than restarting its scheduler or replacing inconvenient runs.

Physical records also preserve a concrete [body-state limitation](PHYSICAL-STATE-SEMANTICS.md): the invalid head row aliases a hind foot. Food sensor/outcome body IDs resolve correctly; rendered mesh geometry avoids that row. Cached sensor poses and geometry reconstructed from saved generalized coordinates describe different integration stages. Neither numerical replay nor browser playback independently reruns physics.

## Complete the registered BabyLM comparison

Continue the existing [training protocol](BABYLM-PROTOCOL.md) and [evaluation declaration](BABYLM-EVALUATION.md), without changing sources, exposure or the twelve-condition inventory. All six 10M fits and 100M FLM seed 42 have completion audits; the remaining five 100M fits and complete-study evaluation are pending. Validation losses choose checkpoints and are not substitutes for held-out comparisons.

After all twelve fits complete, audit and freeze every checkpoint selection before test inference. Report source-component codelength and the declared overlap-filtered analyses with exact text denominators. Retain fixed-prompt continuations, repetition and failed outputs. Compare FLM, GRU and transformer under the declared exposure; separate data-volume effects from architecture and capacity changes. Publish updated model/data cards, figures, papers and standalone artifacts. Historical [resume verification](../reports/babylm/resume-readiness.json) records restoration and exposure checks, not a promise of bitwise multithreaded retraining.

## Next inferential priority: compare selection rules

The [operational KC-centered rule](CIRCUIT-SELECTION.md), motivated by [annotation reconciliation](PUBLISHER-ANNOTATIONS.md) and [pathway accounting](SELECTION-PATHWAYS.md), defines eight candidates. Each includes a full declared seed population and selected partners, without silently truncating the seeds. These remain incomplete operational hypotheses, not certified functional circuits. Five-contact visual candidates retain 81–85% of seed-KC contacts but cut about 89% of whole-subset incoming contacts. Current fast dynamics also exclude zero-sign modulatory outputs; merely selecting those neurons does not restore biological plasticity.

The [coordinator](SELECTION-LANGUAGE-COORDINATOR.md) requires complete groups: the candidate, one contact-ranked selection, three uniform draws and three superclass/side/sign-stratified draws, each with measured wiring plus three rewires and training seeds 42/43. That is **64 fits per chosen group**. The 512 callable graph/seed combinations are not a chosen 512-fit study. Node count matches within groups; density and parameter allocation can differ across selectors. Within-subset rewiring tests topology at fixed allocation separately from selection.

Proceed in this order:

1. Run the [actual cost pilot](SELECTION-TIMING-PILOT.md) after the priority trainer exits. Its shared BPTT/AdamW update path passed 128 tiny synthetic updates across all 64 originals; those checks are not measured corpus throughput. Account for omitted overhead and unmeasured rewire/seed cost proxies.
2. Choose complete groups using recorded anatomical rationale and measured cost, before any selection-language score. Freeze the budget, common exposure, source identities, validation choices and [held-out comparison policy](SELECTION-LANGUAGE-TEST.md). No official group, budget, pilot result or fit exists yet.
3. Fit and evaluate every registered condition. Report family means and individual selector/rewire outcomes, including failures, with conditional uncertainty and allocation differences visible. Neither a win nor a loss certifies intact biological function.
4. Use the result to decide whether compartment-specific selection, explicit boundary drive or separately modeled modulation warrants another declared study. Do not assume that changing the subset guarantees improvement.

## Prepared follow-up work and remaining objectives

| Area | Ready now | Still required before a result |
|---|---|---|
| Alternative language learning | [Verified BabyLM inputs](LANGUAGE-LEARNING-INPUTS.md), BPTT/fixed-core/eligibility/no-history updates, a [coordinator](LANGUAGE-LEARNING-STUDY.md), validation selection and [all-eight-gated scorer](LANGUAGE-LEARNING-TEST.md). | Actual [cost pilot](LANGUAGE-LEARNING-TIMING.md), official protocol/budget/identity, eight corpus fits and complete evaluation. No official learning-rule language scores exist. |
| Symbolic instruction transfer | [SCAN data/codec/scoring primitives](INSTRUCTION-TRANSFER.md), [36 prepared source conditions](SCAN-CONDITION-PREPARATION.md), resumable fitting and a [timing command](SCAN-TIMING-PILOT.md). | Actual pilot, downstream budget, immutable whole-study coordinator and all-condition test gate, then official fits and strict held-out generation. No neural SCAN transfer result or physical instruction execution exists. |
| Food-task adaptation | Audited sensors, scripted and unadapted core references, and [fixture-tested action-readout updates](FOOD-READOUT-LEARNING.md) with frozen sensory/core parameters. | A declared matched adaptation study, episode/action budget, untouched evaluation layouts and reversal/retention phases. No physical adaptation fit or learned food result exists. |

These preparations do not displace selection. The later program still includes:

- **Broader data and capability.** After the registered comparison and selection study, use bounded, licensed, reproducibly selected corpora with document-level provenance, duplicate audits and train-only preprocessing. Measure grammar, generation, retention and domain adaptation. Keep teacher-assisted curation labeled; raw audio, general conversation and code execution require distinct datasets and evaluations.
- **Capacity and implementation scaling.** Vary neurons, retained edges, lexical width and parameter count separately, preserving body-ID mappings and explicit boundary losses. Measure forward/backward cost, memory and sustained behavior before long runs. Larger subsets or a whole-connectome model remain research stages, not assumed quality or energy gains.
- **Alternative biological learning.** Compare forward-time and reward-modulated rules with BPTT, fixed-core and history controls. Verify claimed exact gradients and identify approximations; measure loss, sample efficiency, retention, compute and plasticity memory. Harder delay, reversal and interference tasks remain useful. Spiking and rate models need separate declarations and time semantics.
- **Controlled language-to-action transfer.** Preserve SCAN memberships, repeated-row weights and malformed outputs as errors. Separate the effects of lexical pretraining, recurrent changes and a newly trained action interface. For food, distinguish learned readout, core adaptation, sensory association, physical approach and ingestion; the current references establish no feeding. Use shared initial situations, scripted controls and adequate adapter/gait seeds. Rhythm/game-like behavior remains a later controlled objective, without implying integration with or success at a commercial game.

The [continuing research program](RESEARCH-PROGRAM.md) and [food-response plan](FOOD-RESPONSE-PLAN.md) retain broader questions and source discussions. Completion of the browser, compact benchmarks, structural exports or physical replay does not complete these remaining experiments.

## Delivery and verification at each stage

- Declare the experiment before fitting; retain prior protocols, failed attempts and every registered outcome. Use new identities and output directories for independent replications.
- Keep source licenses, data and graph provenance, exposure, checkpoint selection and test denominators reproducible. Distinguish fixture readiness, recorded arithmetic, independent inference and independent retraining.
- Check restored checkpoints, exports and displayed neural/body state. Test affected browser flows, accessibility, downloads and failure handling; retain dated deployment evidence.
- Maintain the README, model/data cards, figures, working papers and standalone reproduction links. Keep language learning, browser adaptation, sensory decisions and physical simulation distinct.
- Make sequential, descriptive commits, including delegated work, and verify deployed artifacts. Preserve repository visibility, raw-corpus exclusions and component licenses.
