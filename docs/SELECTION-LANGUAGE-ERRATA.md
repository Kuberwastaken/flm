# Selection-language v1: protocol wording correction

Kuber Mehta · 12 September 2026, during the first registered fit and before
selection-language validation or held-out results.

The frozen [protocol](SELECTION-LANGUAGE-PROTOCOL.md) incorrectly says that
the training boundary mask differs from the completed BabyLM comparison.
Both training paths include boundary targets after the 16-position warmup.
The old `flm/language_train.py` training loop applies cross entropy to all
`y[:, 16:]`; the selection runner requires `score_boundaries: true`, and
`flm/language_learning_train.py` therefore masks only warmup positions.
The separate evaluation paths exclude boundary targets.

The difference is in exposure (3,000 versus 12,000 updates), graph selection
and the registered machinery, not this training mask. The comparison remains
a limited-exposure selection screen. No implementation, setting, graph,
checkpoint, input or frozen identity changes. The original protocol is retained
byte-for-byte so its source binding remains verifiable. This correction changes
the interpretation of one sentence, not the experiment.
