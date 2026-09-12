# Selection study: Mac execution amendment

Kuber Mehta · 12 September 2026. Authorized by the request to run training on the MacBook. Recorded before any official Mac fit, validation selection or held-out outcome. This is an execution and native-initialization amendment, not the original preregistration.

## What changes

Run the **entire 128-condition matrix freshly on the Apple M5 Pro CPU**, with a dedicated Python 3.10 environment: PyTorch 2.8.0, NumPy 2.2.6, SciPy 1.13.1, tokenizers 0.22.2 and requests 2.32.5. The runtime profile records the exact Python patch, OS, package builds and adapter hash. The Windows runtime reports PyTorch 2.8.0+cpu. Neither GPU/MPS nor mixed precision is used.

The native ARM probe reproduced all graph/corpus bindings but changed all 128 initial-state, learned-parameter and non-edge-parameter hashes. Identical seed numbers therefore do not mean identical cross-platform initial tensors. **Do not resume Windows optimizer/checkpoint state on the Mac or combine completed Windows fits with fresh Mac fits.** Preserve the original Windows attempt and its immutable identity. It is superseded for this decision by a separate, complete Mac cohort; its stopped runs are an operational interruption, not a negative scientific result.

The two disposable ten-update synthetic-token probes measured about 0.043 seconds/update on the Mac after discarding the first two updates. They did not load BabyLM, validation or test payloads. The contemporaneous Windows probes ran alongside the existing trainer, so their roughly 0.57–0.65 seconds/update are not a controlled platform speed comparison. Neither probe estimates held-out runtime or establishes convergence, final numerical equivalence or language quality. Preserve their full reports, including the initialization differences. Re-estimate the schedule using completed Mac fits.

## What is retained

Keep both original groups, all eight selectors, all three rewires, both seeds and all 128 fits. Preserve graph arrays, signs, node/edge allocations, data bytes, tokenizer, training document order, 3,000 updates, batch 16, sequence 96, warmup 16, float32, four CPU threads per fit, optimizer settings and 500-update checkpoint intervals. Native initialization must still match across rewires within each selection/seed, and non-edge initialization across equal-size selectors. No checkpoint or condition is chosen based on emerging loss.

Retain the original all-fits-before-validation and all-selections-before-test gates, all six checkpoint candidates, official and overlap-filtered analyses, source breakdowns, every paired comparison and the fixed bootstrap policy. All 26 original numerical source files stay byte-identical. The old pilot stays unchanged and explicitly historical; its Windows initialization and timing are not relabeled as native Mac observations.

Apply the **same** dated [continuation rule](anatomical-prior-decision.md) to the complete Mac cohort. The 0.005 BPB margin and all four-cell, interval, twelve-individual-contrast and two-analysis requirements do not change. Pass permits at most one bounded independent confirmation; valid fail stops anatomical subset searches. Incomplete/invalid evidence cannot pass or become scientific failure. This execution amendment is not the independently declared confirmation allowed after a pass.

## Separate identity and execution

The [runtime adapter](selection-mac-runtime.py) verifies the original identity, historical protocol/pilot, all frozen numerical sources, graph archive, corpus, validation panel and test metadata. It reconstructs every native initialization and checks that only the three initialization hashes differ within condition bindings. It also checks unchanged training-document hashes, parameter counts and all matching constraints, and binds its own source and this amendment into the new identity. It does not spoof version strings or alter the historical pilot.

The adapter explicitly redirects the original coordinator/scorer's study paths and context verifier. The original fitting, checkpoint recovery, validation selection, held-out scoring and statistical functions are reused unchanged. This additional execution source is part of the amended method and must be retained in the audit; the 26 original source hashes alone are no longer a complete description of orchestration.

New identities and reports live in `reports/selection-language/mac-v1/`; fresh fits in `runs/selection-language-mac-v1/`; evaluation caches in `runs/selection-language-mac-evaluation-v1/`. The original `reports/selection-language/study-identity.json`, prospective interpretation record and Windows run directory remain intact. The new identity binds the original identity and this amendment. Report the Mac lineage explicitly instead of representing it as the original Windows execution.

```sh
python scripts/selection_mac_runtime.py initialize
python scripts/selection_mac_runtime.py train
python scripts/selection_mac_runtime.py select
python scripts/selection_mac_runtime.py test
```

The `run` operation executes all stages serially with failure propagation. A supervised background process may use it with a log and a process-bound sleep inhibitor on AC power. Verify its actual process and first committed checkpoint before stopping the Windows supervisor. Stop and retain the old attempt; do not let the old automatic queue perform redundant training or selection. Follow-up work uses the Mac paths and adapter rather than restarting the Windows command.
