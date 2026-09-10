# FLM implementation plan

Started 10 September 2026. The repository starts empty; the research workspace already contains a verified MaleCNS extraction and separately cloned public reference material.

## Deliverable

A working, documented FLM implementation with a trained compact checkpoint, reproducible data/graph tooling, held-out evaluation, browser inference and learning, an anatomical activity view, and ChatFLM deployed to `flm.kuber.studio` if the account supports Pages. Larger whole-graph models must be supported or explicitly remain a documented research stage, not implied by the compact release.

## Sequential milestones

1. [x] Establish repository naming, work rules and research direction.
2. [x] Record scientific sources and formal architecture/data decisions.
3. [x] Build verifiable graph acquisition/selection with source-ID provenance.
4. [x] Acquire licensed speech transcripts and audit train/validation/test isolation.
5. [x] Implement trainable recurrent core, slow state and controlled adaptation.
6. [x] Verify gradients, causality, reset behavior and checkpoint reproducibility.
7. [ ] Train bounded compact experiments and compare real/rewired/no-recurrence models.
8. [x] Export a browser checkpoint and verify Python/JavaScript parity.
9. [x] Build ChatFLM: generation, conversation storage, model controls, learning and actual neural visualization.
10. [ ] Add model/data cards, evaluation results, installation and contribution documentation.
11. [ ] Verify desktop/mobile UI, keyboard accessibility, downloads and failure handling.
12. [x] Configure Pages/CNAME, deploy, verify the live domain and commit final evidence.

## Improvement hypotheses

- Diverse bounded time constants can retain useful context with low recurrent-state cost.
- Restricted, explicitly reset local adaptation can learn a user's recent vocabulary without retraining all edges.
- The measured wiring may improve sample efficiency relative to a degree/sign-matched randomized graph. This is a hypothesis to test, not the product tagline.
- Compact browser inference makes experimentation available without a hosted model API.

The first release trains on transcript text, not raw audio. It will be a small completion model, not an instruction-tuned assistant with world knowledge. The interface should make the model's actual capability clear without burying the experience in implementation detail.


## Current evidence and remaining work

WikiText-2 raw is now the primary standard-corpus experiment, with 600/60/60 articles and a frozen 4,096-token train-only vocabulary. The compact lexical model and AMI dialogue model both run in the browser. Independent PyTorch/browser parity, worker integration and causal/resume checks pass. The Research view compares actual matched-update validation curves and unedited same-prompt outputs for FLM, GRU and transformer. Pages and HTTPS are configured and verified at the custom domain.

All six WikiText runs completed 6,000 updates. Frozen validation-selected checkpoints have been scored on complete test articles: two-seed mean FLM 1.9744, GRU 1.9049 and transformer 1.8767 bits/byte. Paired article intervals favor the baselines. Runtime/state measurements and both-seed acute mechanism diagnostics are complete. Keep the simpler n-gram reference separate from parameter-matched models. BabyLM 10M/100M acquisition and encoding are verified; its queue is paused at saved checkpoints while the language topology controls run. Real physical walking controls and a fixed BLiMP grammar diagnostic are also implemented.

Remaining work centers on the continuing program: complete the twelve BabyLM
runs and component-level held-out evaluation; train matched rewired and disabled
mechanism controls before attributing benefits to anatomy; strengthen the
completed forward-learning comparison with harder memory tasks; test language-
to-control transfer and online motor learning; and profile larger
configurations before committing long runs. Extend grammar diagnostics and
generation analysis without tuning on their evaluation scores. Updated model
cards, figures, papers, browser checks and deployment evidence accompany each
completed stage. Full-connectome language training and learned motor behavior
remain unestablished by the compact release.

BabyLM's full evaluation and generation path is implemented and tested before
its test losses are read. All twelve runs must complete before checkpoint
selection is frozen. Complete-block batching preserves causal state and exact
text counts; atomic batch caches resume long evaluations. The report separates
six source components and the predeclared overlap-filtered analysis, with paired
block intervals. Twelve original prompts and two sampling seeds fix 288 unedited
continuations. A dated Research-page snapshot exposes progress and permits only
common-update comparisons. `flm.babylm_pipeline` chains the stages when the current
training writer has stopped; it must not be launched alongside that writer.

The first forward-learning comparison is complete: five rules, three seeds,
7,200 episodes per run, with shared initialization and sensory streams. Full
BPTT wins the final long-delay diagnostic; supervised eligibility exceeds its
no-history control but does not beat the fixed-core mean. Forty separately
simulated physical replays verify that learned high-level choices drive the
expected turning command, retaining wrong task choices. The four-page research
note, all panel predictions, neural states and trajectories are published with
the protocol. This establishes neither an anatomical advantage nor learned gait
or language-to-motor transfer. The subsequent 60-run wiring/context study is now
complete, with shared initialization, degree/sign/self-edge-preserving controls
and exact-gradient diagnostics. Its results are mixed across tasks and learning
rules; they establish no general anatomical advantage. The subsequent online
pose-feedback assay is described below.

The current priority is the language topology study, declared before fitting
its controls. Three null graphs are crossed with both original WikiText seeds;
two further measured-graph models retrain without slow state. Every condition
uses the original 6,000-update exposure. The historical/current trainers match
exactly on initialization, ten updates, sampling and validation replay for both
seeds. The bounded two-worker queue is running. Test evaluation refuses to begin until
all eight new runs finish and all ten checkpoint selections are frozen. Only
then resume the larger BabyLM queue, saved at FLM 12,000 and GRU 6,500 updates.

Recovery uses `python -m flm.language_topology_queue --workers 2 --score` after
verifying there is no existing writer. `python -m flm.language_topology_test snapshot` publishes
checkpoint-backed validation progress; `python -m flm.language_topology_test
score` freezes and scores the completed comparison. These commands continue
this hash-bound experiment and deliberately reject changed data, graph,
numerical source or checkpoint identities. Independent replications need their
own declared identities and freshly trained references, not edits to these
frozen records.

The original serial process had an attached finalizer. The bounded scheduler
replaces that arrangement, holding an OS queue lease and allowing at most two
separate condition writers. Each retains the prescribed four threads and fixed
budget; a memory guard can defer the second worker. Its `--score` option invokes
the strict selection gate, complete test scoring and dated snapshot after all
training children finish. Do not launch an additional finalizer or legacy queue.
See [the operational transition and recovery instructions](LANGUAGE-TOPOLOGY-OPERATIONS.md).
Completed results still require scientific review and browser validation before
their next deployment.

The exact four language graphs now have a public structural audit, per-node
statistics and a licensed download bundle. It recomputes degree/sign/weight
constraints, graph overlaps, reciprocity and components without reading model
losses. The Research view exposes all controls and a common-order matrix figure.
This documents the null intervention; it neither changes the registered graphs
nor resolves the pending language outcome.

All six completed WikiText FLM/GRU/transformer checkpoints now have a compact
standalone inference release. Model-only exports preserve the exact tensors and
fixed buffers and reproduce all 24 published continuations. A fresh extraction
also runs each of the six models through its CLI with the original fixed prompt.
The release includes the shared tokenizer, graph, runtime, protocol and file
hashes; access to the private repository is unnecessary for inference. This is
reproduction work alongside the topology queue, not a new fitted comparison.

The online pose-feedback assay is complete. All 27 conditions and an exact
repeat passed independent causal replay. The three trained cue controllers
match the scripted reference's body trajectory in every scenario; live pose
improves their tracking relative to frozen pose. There are seven distinct
physical trajectories across the cohort, not 27 independent motor skills.
The four-page note, scenario figures, neural state, video and standalone record
archive retain all outcomes and make the engineered wrapper explicit. A fresh
archive extraction passes its own NumPy-only audit without the repository.
No language checkpoint, frozen topology input or language schedule changed.
Language-to-control transfer and online motor learning remain later studies.

Five of the eight new language controls have now completed training. Null
graphs 101, 103 and 107 at training seed 42 have validation loss 1.9072, 1.9083
and 1.9086 BPB, respectively, versus 1.9097 for the measured reference. These
small differences favor the nulls on validation only; the full test comparison
remains gated. The retrained no-slow-state model at seed 42 has validation loss
1.9195 BPB versus 1.9097 for the corresponding full model; this is one seed's
validation comparison. Null graph 101 at seed 43 has now completed with 1.9092
validation BPB versus 1.9107 for its measured reference. The scheduler continues
null graphs 103 and 107 with training seed 43.

The reviewer-requested language computation controls now have separate,
fixture-tested mechanism definitions: fixed recurrent dynamics, no lateral
recurrence, and no temporal state. They do not change the frozen study or start
training. A declared follow-up harness, checkpoint restoration and selection
gate remain to be implemented after the topology result is reported. See
[the control definitions and limits](LANGUAGE-CORE-CONTROLS.md).

Preparation for language-to-action transfer now uses SCAN's three canonical
splits. All 20,910 command interpretations, official memberships, repeated-row
weights and causal target masks are audited. A one-token-per-action codec uses
the existing vocabulary and keeps complete examples within 72 tokens, avoiding
an accidental context advantage over the 96-token transformer. The research
note separates planned neural fitting from data checks and later physical
execution. No new trainer is launched; topology and then BabyLM retain priority.
See [the data and transfer design](INSTRUCTION-TRANSFER.md).
