# FLM implementation and research plan

Started 10 September 2026; status revised 11 September 2026 (Asia/Kolkata).

## Objective and scope

Develop and evaluate FLM as a language model trained from scratch, with reproducible anatomical graph construction, documented data, matched comparators and ChatFLM at [flm.kuber.studio](https://flm.kuber.studio). The compact browser release is working. It is the starting point for larger-data experiments, capacity scaling, alternative learning rules and controlled language-to-action transfer, not completion of that broader objective.

The central question remains whether measured neural wiring or its associated computation supplies a useful prior for prediction. Benefits must be demonstrated against appropriate controls. A negative result is part of the evidence, not a reason to hide a comparator or redefine the experiment. The current model is a small text completion system; neither the available results nor its articulated fly view establish an intact simulated brain, conversational competence or language-driven motor skill.

## Completed evidence

| Work | Verified outcome and boundary |
|---|---|
| Graph and compact architecture | Acquisition, anatomical IDs, graph selection, recurrent/slow-state implementation, causal inference and export are implemented. The [subset audit](SUBSET-AUDIT.md) documents the computational selection of 1,024 neurons and its large cut boundary; it is not an intact circuit. |
| WikiText baseline comparison | All six runs and complete test evaluations finished. Two-seed mean bits/byte: FLM **1.9744**, GRU **1.9049**, transformer **1.8767**; lower is better. FLM trails both matched comparators. See [results](../public/research/test-results.json) and the [protocol](WIKITEXT-PROTOCOL.md). |
| Language topology and slow-state controls | All eight new fits completed; all ten checkpoint selections were frozen before scoring the new controls. The [complete report](../reports/language-topology/summary.json) demonstrates **no anatomical advantage in this selected subset and setup**. |
| Actual recurrent-core dynamics | The [seven-condition pulse diagnostic](LANGUAGE-DYNAMICS-FINDINGS.md) measured changes in the two completed WikiText FLM cores and acute parameter swaps. All cases replay exactly in the recorded environment. It adds no fitting, corpus exposure, language score or behavioral conclusion. |
| Sensory learning | The initial five-rule comparison and [60-run wiring/context extension](WIRING-RESULTS.md) are complete. Effects differ across tasks; sensory results do not establish language benefits. |
| Physical feedback | Forty earlier choice replays and the [27-condition live/frozen-pose assay plus repeat](CLOSED-LOOP-REPRODUCTION.md) are complete. All three trained controllers match scripted body paths in each scenario. Language checkpoints are not used, and no gait or online motor learning occurs. |
| Reproducible product and releases | ChatFLM provides browser inference, separately scoped adaptation, conversation storage and actual inference-state visualization. The six-model baseline bundle, ten-model topology bundle, five working papers, component licenses and release audits are published. See the [README](../README.md), [browser verification](BROWSER-VALIDATION.md) and [topology deployment evidence](../reports/language-topology/final-release.json). |

The topology contrasts favor the rewired graphs: measured-minus-rewired test loss averages **+0.001559 bits/byte**, with all six point estimates positive. Three of six conditional article intervals include zero. The contrasts use three null graphs and two training seeds, sharing two measured references; they are not six independent replications. This exploratory extension does not show that anatomy is generally harmful. The separate retrained no-slow comparison favors retaining slow state by **0.010999 bits/byte** on average. The design does not estimate a topology-by-slow-state interaction. Preserve the [frozen declaration](LANGUAGE-TOPOLOGY-PROTOCOL.md), selections, sources and complete records.

The pulse diagnostic uses the same completed language references, not the new computation controls below. Its zero-drive fast spectral radius changes from 1.0207 initially to 0.9912/0.8722 after training. Those local measurements and finite-horizon pulse responses establish neither global stability, useful memory capacity, improved language accuracy nor transfer to a fly's behavior.

## Current priority: complete the language computation study

The separate [protocol](LANGUAGE-CORE-PROTOCOL.md), [source identity](../reports/language-core/identity.json), trainer, restoration checks and test gate are implemented and frozen. Six new fits cross three controls with seeds 42/43, reusing the two completed full-FLM references:

| Control | Question and limitation |
|---|---|
| Fixed dynamics | Freeze recurrent edges, gain and time constants; train the lexical interface. Does optimizing the recurrent core help? This is not a readout-only reservoir or a match in trainable parameter count. |
| No lateral recurrence | Remove communication along graph edges while retaining fast leak and slow state. This model still has temporal memory. |
| No temporal state | Reset both states for every token, including within training chunks. Does carrying history improve on a learned current-token predictor? |

The [dated runtime-progress snapshot](../reports/language-core/runtime-progress.json) lists verified completed fits and the current active condition. Consult that checkpoint-backed record for current progress instead of treating a step count in this plan as live status. No new computation-control test scores are available. The [mechanism and parameter card](LANGUAGE-CORE-CONTROLS.md) distinguishes allocated, trainable, frozen and gradient-connected parameters.

Finish and verify all six fits before freezing all eight selected checkpoint identities and decoding any new-control test likelihood. Score every test article, report all declared paired contrasts and retain failures and conflicting outcomes. Article intervals are conditional on the fitted models; these exploratory comparisons do not estimate generalization across datasets or establish anatomical causality. Exact single-thread and bounded four-thread reference checks passed; four-thread training is not promised to reproduce bitwise across executions.

Follow the protocol's queue and recovery instructions, with one training writer per output directory. Do not restart the completed topology scheduler, alter frozen numerical sources, or add another writer to an active run. Review the full report, inference exports and browser presentation before publishing conclusions. Then resume the registered BabyLM comparison.

## Next: resume the larger-data comparison

BabyLM remains paused during the computation study. Its 10M/100M training corpora, common evaluation partitions, shared train-fitted tokenizer and overlap audit are prepared; one of twelve training runs has completed. The saved runs must continue under the existing [training protocol](BABYLM-PROTOCOL.md) and [evaluation declaration](BABYLM-EVALUATION.md), not be replaced by the small-corpus mechanism study.

Complete all twelve runs and freeze selection before the declared test evaluation. Report source-component codelength and overlap-filtered analyses with exact text denominators. Retain all fixed-prompt continuations, repetitions and unsuccessful outputs. Compare FLM, GRU and transformer at the declared exposure, and distinguish data-volume effects from architecture or capacity changes. Update the [data cards and acquisition status](DATA-STATUS.md), model cards, figures, papers and public release with the complete results.

## Remaining research after these controls

1. **Broader data and evaluation.** Extend the mixed-data study to bounded, licensed corpora and stronger grammar, generation and adaptation evaluations. Preserve source identities, document-level splits, duplicate audits and train-only preprocessing. Keep any teacher-assisted data curation separately labeled. Raw audio learning, broader conversational ability and code execution are distinct objectives, not properties of the present text models.
2. **Capacity and implementation scaling.** Profile forward/backward throughput and memory before long runs. Vary neuron count, retained edges, lexical width and parameter count separately, retaining anatomical source IDs and measuring boundary loss. Match exposure and report costs against GRU/transformer comparators. Larger subsets and whole-connectome language models remain research stages; checkpointing, streaming evaluation and explicit compute budgets must accompany them.
3. **Alternative biological learning mechanisms.** Extend the completed sensory comparisons to harder delayed-memory, reversal and interference tasks before applying viable forward-time or reward-modulated eligibility rules to language. Compare with BPTT, a fixed core and suitable history controls; distinguish exact derivatives from approximations, and measure accuracy, retention, compute and plasticity memory. Rate and spiking models require separate declarations. Neither anatomical wiring nor an eligibility trace proves biological fidelity.
4. **Controlled language-to-action transfer.** [SCAN data preparation](INSTRUCTION-TRANSFER.md) is complete for the simple, length and added-primitive splits, including all 20,910 command interpretations and the reversible action codec. The shared masked-loss, strict-generation and resumable-fitting primitives are fixture-tested across all three architectures; they are preparation, not an instruction benchmark. No neural transfer fits or physical instruction execution have started. Freeze the downstream training budget, source identities and evaluation gate before fitting the planned matched initial-versus-language-trained comparisons. Preserve official split memberships and repeated-row weights; retain malformed and incorrect generated actions as errors. Separate any benefit of lexical pretraining, recurrent changes and a newly learned action interface.
5. **Grounded behavior and online motor learning.** After symbolic transfer, compare fixed, language-trained and task-adapted cores through the same learned sensory/action interface, situations and action budget. Keep instruction accuracy, physical execution, gait and online adaptation as separate outcomes, with adequate scripted controls. The existing body wrapper already solves its simple path tasks; a future task must measure what learning adds. Rhythm/game-like tasks remain possible later experiments, without implying integration with or success at an existing commercial game.

The [continuing research program](RESEARCH-PROGRAM.md) records the wider questions and sources. New studies must follow existing compute priorities rather than competing with the active queue. Completion of the compact release, topology report or diagnostic does not complete the larger-data, scaling, alternative-learning or transfer work listed here.

## Delivery and verification at each stage

- Declare the next experiment before fitting, preserve prior protocols and retain every completed observation. Independent replications need separate identities and output directories.
- Keep data provenance, licenses, graph scope, exact exposure, checkpoint selection and test denominators reproducible. Publish actual positive, null and negative results with their uncertainty limits.
- Verify implementation, restored checkpoints, exports and any displayed inference state. Test the browser on desktop/mobile, keyboard access, downloads and failure handling when relevant changes land; retain dated evidence rather than treating checks as a permanent guarantee.
- Maintain the README as the entry point, with model/data cards, computational figures, working papers and standalone reproduction artifacts. Distinguish browser adaptation, language learning, sensory learning and physical simulation in the interface and documentation.
- Make sequential, descriptive commits, including delegated work, and verify the deployed artifacts at `flm.kuber.studio`. Keep repository visibility, raw-corpus exclusions and component licenses unchanged.
