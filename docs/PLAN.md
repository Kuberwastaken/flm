# FLM implementation plan

Started 10 September 2026. The repository starts empty; the research workspace already contains a verified MaleCNS extraction and separately cloned public reference material.

## Deliverable

A working, documented FLM implementation with a trained compact checkpoint, reproducible data/graph tooling, held-out evaluation, browser inference and learning, an anatomical activity view, and ChatFLM deployed to `flm.kuber.studio` if the account supports Pages. Larger whole-graph models must be supported or explicitly remain a documented research stage, not implied by the compact release.

## Sequential milestones

1. [x] Establish repository naming, work rules and research direction.
2. [ ] Record scientific sources and formal architecture/data decisions.
3. [ ] Build verifiable graph acquisition/selection with source-ID provenance.
4. [ ] Acquire licensed speech transcripts and audit train/validation/test isolation.
5. [ ] Implement trainable recurrent core, slow state and controlled adaptation.
6. [ ] Verify gradients, causality, reset behavior and checkpoint reproducibility.
7. [ ] Train bounded compact experiments and compare real/rewired/no-recurrence models.
8. [ ] Export a browser checkpoint and verify Python/JavaScript parity.
9. [ ] Build ChatFLM: generation, conversation storage, model controls, learning and actual neural visualization.
10. [ ] Add model/data cards, evaluation results, installation and contribution documentation.
11. [ ] Verify desktop/mobile UI, keyboard accessibility, downloads and failure handling.
12. [ ] Configure Pages/CNAME, deploy, verify the live domain and commit final evidence.

## Improvement hypotheses

- Diverse bounded time constants can retain useful context with low recurrent-state cost.
- Restricted, explicitly reset local adaptation can learn a user's recent vocabulary without retraining all edges.
- The measured wiring may improve sample efficiency relative to a degree/sign-matched randomized graph. This is a hypothesis to test, not the product tagline.
- Compact browser inference makes experimentation available without a hosted model API.

The first release trains on transcript text, not raw audio. It will be a small completion model, not an instruction-tuned assistant with world knowledge. The interface should make the model's actual capability clear without burying the experience in implementation detail.
