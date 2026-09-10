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

All six WikiText runs completed 6,000 updates. Frozen validation-selected checkpoints have been scored on complete test articles: two-seed mean FLM 1.9744, GRU 1.9049 and transformer 1.8767 bits/byte. Paired article intervals favor the baselines. Runtime/state measurements and both-seed acute mechanism diagnostics are complete. Keep the simpler n-gram reference separate from parameter-matched models. BabyLM 10M/100M acquisition and encoding are verified; its registered training queue is active. Real physical walking controls and a fixed BLiMP grammar diagnostic are also implemented.

Remaining work centers on the continuing program: complete the twelve BabyLM
runs and component-level held-out evaluation; train matched rewired and disabled
mechanism controls before attributing benefits to anatomy; strengthen the
completed forward-learning comparison with harder memory tasks; extend the
learned-choice physics interface to online sensory feedback; and profile larger
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
or language-to-motor transfer. Rewired controls, stronger context-dependent
tasks, learning-signal diagnostics and closed-loop sensory coupling remain open.
