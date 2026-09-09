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

The first-seed transformer has completed 6,000 updates. FLM and GRU are still running, followed by the second seed. No full WikiText test likelihood has been read or reported. The current matched validation result does not favor FLM. Finish both seeds before interpreting the final comparison. Keep the simpler n-gram reference separate from parameter-matched models.

Remaining deliverables: full test scoring with paired article uncertainty; completed fixed-prompt comparisons; standalone runtime/state-memory measurements; graph/slow-state controls before any topology claim; computational figures and LaTeX research/data papers; final model cards and reproducibility audit; browser storage/download/accessibility checks; final deployment verification. The full-connectome language model and learned motor behavior are explicitly future scaling/embodiment stages. They must not be implied by the compact language release.
