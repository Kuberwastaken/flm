# Continuing FLM research program

This extends the compact WikiText release. WikiText is a calibration experiment,
not the endpoint. Preserve its frozen protocol and report before moving on.

## Current priority: isolate the language computation

The language topology study completed on 10 September. All eight new fits
finished before all ten checkpoint selections were frozen and the new controls
were test scored. The [complete report](../public/research/language-topology-results.json)
shows lower loss for all six rewired fits: measured minus rewired averages
+0.001559 bits/byte, with three of six conditional article intervals including
zero. No measured-wiring advantage is demonstrated in this subset and setup.
The retrained slow-state comparison separately favors retaining slow state by
0.010999 bits/byte on average. Six wiring contrasts share two measured references;
the design does not estimate a topology-by-slow-state interaction.

The [declared protocol](LANGUAGE-TOPOLOGY-PROTOCOL.md), source identities and
checkpoint selections remain frozen. The standalone score archive reproduces
all arithmetic without training-code imports. The ten-model inference archive
matches its selected tensors and buffers, all forty source continuations, and
one fresh-archive CLI replay per condition. These checks establish the published
record's reproducibility boundaries, not independent retraining.

The separate [language computation study](LANGUAGE-CORE-PROTOCOL.md) is frozen
and its serial training queue has started. It retrains fixed-dynamics,
no-lateral-recurrence and no-temporal-state controls with seeds 42/43, against
the two existing full-FLM references. No new language test scores are available.
All six new fits and eight selections must finish before test scoring.
Its [identity](../reports/language-core/identity.json) records exact shared
initialization and sampled exposure, exact single-thread reference replay, and
bounded four-thread numerical replay. Twenty-one fixture tests cover mechanisms,
checkpoint corruption, interrupted saves, exact single-thread resume and test
gates. Four-thread bitwise trajectory reproducibility is not established.
The larger-data queue stays paused: the first BabyLM FLM run completed 12,000
updates and the GRU is saved at 6,500. Resume its registered comparison after
the language computation controls; do not replace the broader data/domain study
with this small-corpus result.

The [completed 60-run cue/context study](WIRING-RESULTS.md) gives mixed results
and cannot answer the language claim. Keep its evidence separate from language,
browser adaptation and physical motor control.

The [subset audit](SUBSET-AUDIT.md) explains the computational selection of
1,024 neurons and quantifies the large boundary loss: 79.42% of incoming and
73.96% of outgoing raw contacts. No intact learning circuit is established.
Interpret the topology result in that restricted setting. The next language
causality questions are [fixed recurrent dynamics, no lateral recurrence and
no temporal state](LANGUAGE-CORE-CONTROLS.md); their language fits are now under
way, with results pending. They must not alter the completed
topology study. SCAN remains prepared and deferred.

## 1. Preserve the completed shared language reference

Both seeds of FLM, GRU and the compact transformer are complete. Validation-selected
checkpoints were scored on every test article, with paired article intervals,
fixed-prompt samples, actual runtime and state storage published. Mean test
bits/byte are 1.9744 for FLM, 1.9049 for GRU and 1.8767 for the transformer;
lower is better. Keep that negative comparison visible. The
[six-model standalone release](INFERENCE-BUNDLE.md) reproduces the published
continuations without the training corpus or private repository.

Acute recurrence and slow-state diagnostics remain separate from retrained
ablations. The completed degree/sign-matched rewiring experiment demonstrates
no anatomical advantage here. Preserve its [dated snapshot](../public/research/language-topology-progress.json)
and full score records when moving to the next study.

## 2. Scale data and measure domain differences

The official BabyLM 2026 Strict-Small and Strict corpora have been acquired at
immutable revisions with their common development and test partitions. Their
source-indexed caches, train-fitted shared tokenizer and overlap audit are ready;
see [data status](DATA-STATUS.md) and the [evaluation declaration](BABYLM-EVALUATION.md).
Resume the registered training comparison after the language computation controls.
The six components
cover spoken language, child-directed speech, books, subtitles and simple
encyclopedic text. Audit actual words, UTF-8 bytes, line/document boundaries and
cross-split duplicates rather than assuming the advertised 10M/100M labels are
tokenizer counts. Keep each component identifiable in evaluation. The publisher's
MIT metadata does not replace the underlying components' provenance or rights.

Train shared-tokenizer comparators from scratch on declared subsets and exposure
budgets. Evaluate component-wise codelength, fixed-prompt generation, repetition,
syntactic minimal pairs and adaptation transfer/retention. Compare domain effects
at fixed model size, then capacity effects at fixed data exposure. Use the official
GPT-2-family BabyLM baselines as separately labeled external references when useful;
their parameter counts, tokenizers and training budgets differ from FLM's.

Proceed to a bounded, reproducibly selected web corpus after the mixed-data study.
FineWeb/FineWeb-Edu provide an auditable route to larger pretraining data. Preserve
document identities, URL/domain split policies, exact and near-duplicate audits,
and source-crawl dates. FineWeb-Edu's text is web-derived, but its quality filter
was learned using Llama-generated annotations; it is not a teacher-independent
curation process. Do not silently combine its results with the initial first-
principles experiment. Code-specific training requires a separately licensed and
repository-disjoint corpus, plus execution-based evaluation in a sandbox.

## 3. Compare learning rules, not just wiring

Build a forward-time eligibility-trace learner and a reward-modulated local
plasticity learner alongside BPTT and a fixed-reservoir/readout control. Start with
delayed cue, evidence accumulation, reversal and interference tasks whose targets
and train/test seeds are controlled. Verify mathematical gradients where a rule
claims an exact derivative, and explicitly name approximations where it does not.
Measure success, sample count, retention, computational cost and state/plasticity
memory. Then apply viable rules to the same language data and report losses and
generation even when they underperform BPTT.

Eligibility traces and three-factor plasticity are prior research, not unique to
fly graphs. The anatomical constraint makes specific structural hypotheses
testable; it does not establish that an algorithm is possible only in FLM. Spiking
and rate formulations should remain distinct and receive explicit time units.

## 4. Situations and embodied behavior

The first local-learning study is complete: five rules across three seeds, with
identical starting weights and stimuli within each seed. The final 48-frame
diagnostic gives BPTT 100%, fixed core 86.46%, supervised eligibility 83.33%, and
no-history/reward eligibility 50%. These are repeated diagnostics from one
simple task, not a general ranking. The fixed core's strong result motivates
the completed 60-run context and rewiring extension, whose task-dependent
results are recorded in [the wiring findings](WIRING-RESULTS.md). See
[the declared protocol](LOCAL-LEARNING-PROTOCOL.md) and the
[working research note](../public/research/local-learning.pdf).

A real NeuroMechFly simulation now runs in an isolated, pinned FlyGym/MuJoCo
environment. Its calibrated turning interface and all recorded physical studies
are reproducible. The next transfer experiment should compare a fixed core,
language-trained core and task-adapted core using the same initial situations and
action budget. Learning a sensor-to-action interface, balance or gait requires
separate objectives and controls; the current studies do not learn joint control.

Separate the effect of learning a new motor head from changes in recurrent
dynamics. A rendered pose change is not a learned skill. A language-trained core
controlling an artificial body is not evidence that a biological fly knows words.
Rhythm/cue timing can become a controlled game-like task after the motor interface
works; a real game's integration and score must not be implied by a toy task.

A first learned-choice assay now connects the smaller sensory network to the
calibrated physical command interface. All forty predetermined cases were
simulated independently; recorded heading follows the selected command in every
case, including wrong neural choices. This is an offline high-level decision
followed by physical replay. The two commands yield two reproducible paths;
learning joint control and transfer from language training remain separate
experiments.

The subsequent [online pose-feedback assay](CLOSED-LOOP-REPRODUCTION.md) is also
complete: four fixed sensory checkpoints, live/frozen pose, three waypoint
scenarios, three scripted references and one exact repeat. Its 27 primary
conditions contain 10,800 control frames and 972 delayed decisions. In the
switching-target case, live pose gives the trained controllers a mean absolute
bearing error of 16.27 degrees, versus 100.50 degrees with frozen pose. However,
all three trained methods produce exactly the same physical paths as the
scripted reference in each scenario. The assay establishes that the engineered
feedback interface works; it does not establish a need for learned recurrence
or an anatomical advantage. Ground-truth pose supplies an artificial cue, and
the designed FlyGym controller supplies gait and contact feedback. Language
weights are not used and no learning occurs during this physical evaluation.

Retain every condition, including unsuccessful initial models. The complete
records and standalone causal replay audit are public. Language-to-control
transfer and online motor learning are still open; future experiments must
separate their effects from this already adequate scripted controller.

## 5. Capacity and implementation scaling

Measure forward/backward throughput and memory before committing long training
runs. Scale neurons, retained edges, lexical dimensions and parameter counts
separately; preserve the source-ID mapping. Add checkpoint/resume, streaming data
and bounded-memory evaluation to every larger configuration. A GPT-2-sized
configuration or corpus is not a GPT-2-quality result: report the actual training
exposure and held-out capability. The available machine has a CPU and integrated
graphics, so local studies start with measured, recoverable runs. No paid compute
is provisioned by this plan.

## Sources

- [BabyLM 2026 overview](https://babylm.github.io/) and [guidelines](https://babylm.github.io/guidelines.html).
- [Official 10M corpus](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small), [100M corpus](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict), and [GPT-2-family baseline](https://huggingface.co/BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small).
- [FineWeb-Edu publisher card and curation](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
- [Bellec et al., eligibility propagation](https://www.nature.com/articles/s41467-020-17236-y).
- [Dhiman and Panwar, optic-lobe information processing and eligibility-trace plasticity (2026)](https://www.nature.com/articles/s41598-026-52140-3). This is additional connectome/local-learning precedent; its sensory and energy-proxy regimes are separate from language. The [embedding preparation note](LANGUAGE-ELIGIBILITY-PREPARATION.md) records the scope.
- [FlyGym 2.x documentation](https://neuromechfly.org/) and [installation](https://neuromechfly.org/installation/).
