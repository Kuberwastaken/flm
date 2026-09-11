# Complete language computation comparison

All six new fits completed before the eight checkpoint selections were frozen. The two full-FLM references reuse their previously published test scores. This is an exploratory follow-up on the measured 1,024-neuron subset.

![Every declared fitted-pair effect; panels use different horizontal scales](../public/research/figures/language-core-test.svg)

## Held-out prediction loss

Bits per UTF-8 byte; lower is better. Each model is scored on all 60 test articles.

| Model | Seed 42 | Seed 43 | Descriptive mean |
|---|---:|---:|---:|
| Full FLM | 1.973182 | 1.975583 | 1.974383 |
| Fixed recurrent dynamics | 1.978845 | 1.981514 | 1.980180 |
| No lateral recurrence | 1.979413 | 1.982134 | 1.980773 |
| No temporal state | 2.098829 | 2.098957 | 2.098893 |

## Every declared contrast

Differences are the first model minus the second. Negative favors the first model. Intervals resample paired articles, conditional on these fitted checkpoints; they do not include training or graph uncertainty. The comparisons share articles and references.

| Family | First minus second | Seed | Difference | 95% article interval |
|---|---|---:|---:|---:|
| Primary | Full FLM − Fixed recurrent dynamics | 42 | -0.005663 | [-0.007042, -0.004067] |
| Primary | Full FLM − Fixed recurrent dynamics | 43 | -0.005931 | [-0.006629, -0.005195] |
| Primary | Full FLM − No lateral recurrence | 42 | -0.006230 | [-0.008773, -0.003057] |
| Primary | Full FLM − No lateral recurrence | 43 | -0.006551 | [-0.008918, -0.003656] |
| Primary | Full FLM − No temporal state | 42 | -0.125647 | [-0.133917, -0.117878] |
| Primary | Full FLM − No temporal state | 43 | -0.123375 | [-0.131518, -0.115551] |
| Secondary | No lateral recurrence − No temporal state | 42 | -0.119416 | [-0.127708, -0.111568] |
| Secondary | No lateral recurrence − No temporal state | 43 | -0.116824 | [-0.124713, -0.108930] |

The three primary descriptive means are:

- Full FLM minus fixed recurrent dynamics: -0.005797 BPB.
- Full FLM minus no lateral recurrence: -0.006391 BPB.
- Full FLM minus no temporal state: -0.124511 BPB.

The secondary no-lateral minus no-temporal mean is -0.118120 BPB. No interval on these descriptive means is declared.

## What each comparison tests

- **Fixed dynamics:** the four recurrent parameter groups remain at initialization; the lexical interface is trained through time.
- **No lateral recurrence:** removes graph-mediated communication while retaining independent fast/slow temporal states.
- **No temporal state:** also resets both states at every token. It retains the per-token nonlinear and lexical machinery.

Every condition allocates 600,003 parameter entries. Fixed dynamics freezes 78,179 and trains 521,824. Both recurrence-disabled variants mark 600,003 entries trainable, but 76,131 edge/gain entries are disconnected. These counts do not imply equal effective capacity.

A difference for one of these implemented mechanisms is not an anatomical-topology advantage, equivalence result, or evidence of language-to-behavior transfer. The separate topology study and its negative result remain distinct. Two initializations and one corpus do not establish a general ranking.

## Inspect and reproduce

- [Frozen protocol](LANGUAGE-CORE-PROTOCOL.md)
- [Every selected checkpoint and article score](../public/research/language-core-results.json)
- [Complete records and standalone arithmetic checker](https://flm.kuber.studio/research/language-core-records.zip)
- [All plotted values and selected update counts](../public/research/figures/language-core-test.csv)
- [Frozen selection](../reports/language-core/selection.json)

Figures and tables are generated from the verified complete score report. The arithmetic check does not independently retrain models or rerun held-out inference.
