# Continuing FLM research program

This extends the compact WikiText release. WikiText is a calibration experiment,
not the endpoint. Preserve its frozen protocol and report before moving on.

## 1. Complete the shared language reference

Finish both seeds of FLM, GRU and the compact transformer; select on validation;
score complete test articles; publish paired intervals, samples, actual runtime
and state storage. Keep negative results visible. Acute recurrence and slow-state
diagnostics are separate from retrained ablations. Complete the degree/sign-matched
rewiring experiment before attributing a benefit to anatomical topology.

## 2. Scale data and measure domain differences

Acquire the official BabyLM 2026 Strict-Small and Strict corpora at immutable
revisions, with their common development and test partitions. The six components
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

Establish a real NeuroMechFly physics simulation with a pinned FlyGym version in
an isolated environment. Start with reproducible open-loop and closed-loop walking
controls, then train a documented sensor-to-action interface for cue following,
turning and reward reversal. Record observations, actions, trajectories, contacts,
falls and rewards. Compare fixed core, language-trained core and task-adapted core
using the same initial situations and action budget.

Separate the effect of learning a new motor head from changes in recurrent
dynamics. A rendered pose change is not a learned skill. A language-trained core
controlling an artificial body is not evidence that a biological fly knows words.
Rhythm/cue timing can become a controlled game-like task after the motor interface
works; a real game's integration and score must not be implied by a toy task.

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
- [FlyGym 2.x documentation](https://neuromechfly.org/) and [installation](https://neuromechfly.org/installation/).
