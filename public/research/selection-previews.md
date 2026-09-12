# KC language-model previews

Four completed native Mac checkpoints are available in ChatFLM as **research previews**. Kuber Mehta authorized publishing the trained models while held-out evaluation continues. Training and validation selection are complete; the full anatomical comparison is still pending.

| Preview | Neurons | Literal KCg-d cells | Parameters | Validation BPB |
|---|---:|---:|---:|---:|
| [KC left, seed 42](https://flm.kuber.studio/?model=flm-kc-l-s42#chat) | 487 | 99 | 486,384 | 2.188177 |
| [KC left, seed 43](https://flm.kuber.studio/?model=flm-kc-l-s43#chat) | 487 | 99 | 486,384 | 2.191403 |
| [KC right, seed 42](https://flm.kuber.studio/?model=flm-kc-r-s42#chat) | 540 | 107 | 494,456 | 2.199335 |
| [KC right, seed 43](https://flm.kuber.studio/?model=flm-kc-r-s43#chat) | 540 | 107 | 494,456 | 2.192465 |

Both candidate sides and both seeds are published. These are all four measured candidate models from the frozen study, not a selection of favorable held-out results. Within each fit, the checkpoint was selected by the earliest minimum loss across six saved updates on the fixed validation panel. All four selected update 3,000. The [128-condition selection inventory](https://github.com/Kuberwastaken/flm/blob/main/reports/selection-language/mac-v1/study-selection.json) froze before held-out scoring.

These are **smaller circuit-selection models, not a larger conversational upgrade**. Each received 4,608,000 input-token presentations and 3,840,000 supervised targets from the BabyLM 10M training pool. The earlier primary BabyLM models received 18,432,000 token presentations under a different training boundary policy. Neither corpus size nor a preview release establishes convergence or better chat behavior. The automatic default remains the completed primary BabyLM 100M FLM, seed 43; previews are excluded from that primary ranking.

The core uses measured induced wiring with 15,674 edges on the left-seed candidate or 18,499 on the right-seed candidate. Side names describe the seed population, not hemisphere-pure graphs. Although the candidates retain roughly 81-85% of contacts incident on their seed KCs, they cut about 89% of incoming contacts to their full selected node sets. They do not reproduce intact mushroom-body physiology. See the frozen [protocol](https://github.com/Kuberwastaken/flm/blob/main/docs/SELECTION-LANGUAGE-PROTOCOL.md) and [pathway audit](https://github.com/Kuberwastaken/flm/blob/main/docs/SELECTION-PATHWAYS.md).

## What the browser runs

The downloadable `model.json`, `weights.bin`, `tokenizer.json` and `anatomy.json` under each `/models/flm-kc-*/` directory contain that checkpoint's predictor and anatomical mapping. The brain displays its own 487 or 540 states at the retained neuron positions, with the corresponding cell identities and types. Gray context anatomy is a reference. Generation updates the actual state every token. The articulated typing fly remains an authored visualization, not a learned motor policy.

Chat history, neuron interventions and optional local readout adaptation work as for the primary FLM. These are base next-token models without instruction tuning. The preview label remains visible in chat and continuation modes; incomplete evaluation is not hidden behind a default-model change.

## Reproduction and release checks

The [exporter](https://github.com/Kuberwastaken/flm/blob/main/scripts/export_selection_previews.py) opens the frozen training/selection records, graph archive and selected checkpoints. It does not open held-out text or scores. Checkpoint hashes, complete training exposure, source bindings, tensor shapes and fixed graph buffers are checked before export. Original weights and the running evaluator are not modified.

On a checkout with the recorded selected checkpoints, graph archive and tokenizer, export with `python scripts/export_selection_previews.py`. Existing packages are preserved: use a fresh output checkout for regeneration. The optional `--fetch` mode reads an ignored local Mac connection record and verifies every transferred checkpoint. To generate references without publishing, use `--reference-output <directory>`.

The [export record](https://github.com/Kuberwastaken/flm/blob/main/reports/selection-language/mac-v1/browser-preview-v1.json) binds each package to its source. The [native comparison](https://github.com/Kuberwastaken/flm/blob/main/reports/selection-language/mac-v1/browser-native-parity-v1.json) compares restored Mac and Windows computations at four prefixes through 130 tokens. Maximum observed logit difference was 0.00000334. Browser catalog checks compare all 4,096 logits and fast/slow states; worker checks cover both graph sizes, generation, local learning, interventions and cancellation. The [browser review](https://github.com/Kuberwastaken/flm/blob/main/reports/selection-language/mac-v1/browser-preview-review-v1.json) records generation from all four previews, changing brain renders, both themes and mobile overflow checks. These establish implementation agreement, not language quality or biological advantage.

The unchanged sequential scorer continues evaluating all 128 conditions. Complete official and overlap-filtered results and the [prospective continuation decision](https://github.com/Kuberwastaken/flm/blob/main/docs/ANATOMICAL-PRIOR-DECISION.md) will be published separately. A preview does not pass, bypass or change that scientific gate.
