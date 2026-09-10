# Does recurrent computation improve FLM's language prediction?

Declared on 11 September 2026 (Asia/Kolkata), before fitting these controls.
This exploratory extension follows the completed WikiText baseline and topology
studies. Their test results have already informed the research program. This is
not an untouched-test discovery study, a test of biological learning, or a
topology-by-trainability factorial.

## Conditions and questions

Use the same measured 1,024-neuron graph, train-only 4,096-ID byte-BPE tokenizer,
official WikiText-2 article partitions and language parameterization as the
completed studies. Reuse the two selected full-FLM checkpoints, never their
trained tensors as initialization. Train the following six new models from
scratch, with both training seeds 42 and 43 for every condition:

| Condition | Change throughout training and evaluation | Question |
|---|---|---|
| Full FLM | Existing two matched reference runs | Reference computation |
| Fixed dynamics | Freeze edge gains, both time-constant groups and recurrent gain at shared initialization | Does learning recurrent dynamics help beyond a fixed dynamical feature map? |
| No lateral recurrence | Omit `W h`; retain fast leak and slow temporal state | Does communication along graph edges help beyond independent temporal units? |
| No temporal state | Reset both states for every token, including within each training chunk | Does carrying history help beyond a learned current-token predictor? |

`flm/language_core_controls.py` implements these mechanisms separately from the
completed topology sources. In the memoryless model, `W h` is also omitted:
its input is zero at every fresh state. This produces the full model's
single-token-reset computation, including its fast/slow feature widths and
LayerNorm. The control can learn a nonlinear next-token distribution from the
current token. It cannot use previous tokens. This is distinct from resetting
only at chunk boundaries.

All models allocate 600,003 parameter entries. Fixed dynamics freezes 78,179
entries and trains 521,824. Its embeddings, input projection, LayerNorm and
readout learn, with gradients through time to the lexical interface. It is
neither a readout-only reservoir nor a parameter-matched training comparison.
The two recurrence-disabled models retain 600,003 trainable entries in storage,
but the 76,130 edge gains and one recurrent-gain scalar are disconnected from
their objectives. Report allocated, trainable, frozen, and probe-connected
entries separately. A gradient probe is not an estimate of effective capacity.

For each seed, all four conditions must have identical named initial parameter
tensors and identical post-construction PyTorch RNG state. Graph buffers and
pooling also match exactly. Confirm the new full-model update implementation
against the completed trainer on ten real training-text updates and fixed
validation prefixes. Require exact single-thread parameter/loss replay and
validation NLL. Separately repeat at the historical four training threads,
requiring maximum absolute parameter differences at most 1e-7, loss differences
at most 1e-6, and aggregate/per-article validation NLL differences at most 1e-4.
Retain all measured differences. These numerical bounds are compatibility
checks, not statistical equivalence margins or permissible changes to results.
Those checks fit only disposable full references, not
the six new control runs. Fixture training, corrupted-checkpoint tests and
artificial-token gradient probes are software preflight, not language evidence.

Before freezing this study, a bitwise four-thread preflight failed. A diagnostic
ten-update original/new replay had identical losses and a maximum parameter
difference of 5.82e-11, first appearing in edge gains; the corresponding
single-thread replay was exact. This has not been traced to a specific kernel.
The separate exact and bounded checks above were declared in response to that
numerical diagnostic, before fitting any new condition. Training still uses
the historical four-thread settings. Do not promise bitwise repeatability of
future four-thread trajectories across executions or hardware.

## Fixed training budget and selection

Keep the original protocol: batch 16, sequence 96, first 16 positions excluded
from loss, 6,000 AdamW updates, 9,216,000 presented input tokens per run, learning
rate 0.002 with 100-update linear warmup and cosine decay to 0.0002, weight decay
0.01, global gradient clipping at 1, four CPU threads, float32. Retain the
original AdamW defaults and parameter order, filtering only explicitly frozen
parameters out of the optimizer. Do not omit trainable-but-disconnected groups
from its recorded inventory. Their absent gradients mean they receive no
optimizer moments or weight decay updates.

Use the same sampled windows for a given seed as the completed measured run.
Bind the exact train/validation caches and tokenizer. During preparation,
independently replay the complete sampler sequence and compare its stream
digests, RNG and byte/token exposure at each 500-update boundary with the
completed study's audit. During training and restore, check the live sampler
RNG and accumulated byte/token exposure against that audit at every saved
checkpoint. The training loop does not independently hash its consumed token
stream; stream identity also relies on the frozen sampler code and inputs. State
starts fresh for each sampled training window; BPTT operates within it.

Evaluate the same fixed per-article validation prefixes, totaling the existing
32,768-token quota, at updates 500, 1,000, ..., 6,000. Select the lowest validation
bits per UTF-8 byte; retain the earliest update on an exact tie. The initial
validation score is descriptive and ineligible for checkpoint selection.
No extra tuning, early stopping, condition-specific schedule or test feedback.

Commit numerical sources and this declaration before creating the study
identity. Record source hashes, source commit, all data/graph identities,
reference checkpoints, parameter and optimizer inventories, compatibility
checks, sampled-stream audit, validation denominators and PyTorch version.
Retain the completed topology identity unchanged. Hashing the test cache before
training checks identity without decoding it.

## Saving, interruption and verification

Save all twelve 500-update checkpoints with model, optimizer, sampler and
PyTorch RNG state. A checkpoint manifest is written atomically after the
checkpoint, validation score and interval history; it hashes all three files.
Only a contiguous sequence of committed manifests is eligible for resume or
selection. Ignore incomplete save payloads, then overwrite them by replay from
the latest committed checkpoint. Fail on altered committed bytes, gaps,
different settings or changed study inputs. Do not append repeated training
exposure to a resumed model.

Verify graph and pooling buffers when restoring. For fixed dynamics, construct
the original seeded model and compare every frozen tensor exactly before
loading and before every save. Verify all twelve committed checkpoints at run
completion, including optimizer settings/moments, cumulative validation
selection, sampled exposure, RNG and validation article denominators. Software
fixture tests at one CPU thread must establish exact uninterrupted/resumed trajectories for each mechanism
and rejection of corrupted frozen tensors and incompatible configurations.

Record timing and throughput as descriptive training logs; do not present
them as isolated hardware benchmarks. Record resume boundaries explicitly.

## Test gate, estimands and reporting

All six new runs must finish, all eight selected checkpoint identities must be
frozen, and all declared identities must pass verification before decoding any
new-control test likelihood. Reuse published full-model test scores only with
exact checkpoint, test cache, step, seed and parameter-count agreement.

Score all 60 test articles with the original evaluator: reset per article,
carry native state within an article, chunk size 96, exclude inserted boundary
targets, use exact UTF-8 byte denominators. The memoryless control additionally
resets its own states at every token. Retain per-article NLL, byte/token counts,
aggregate bits/byte and shared-tokenizer perplexity for every run.

Primary effects are full minus fixed-dynamics, full minus no-lateral, and full
minus no-temporal-state BPB, individually for seeds 42/43 and as the arithmetic
mean across those two seeds. Negative favors full FLM. Report all six contrasts
even if they disagree. As a declared secondary contrast, compare no-lateral
minus no-temporal-state for both seeds and their mean. This asks whether
independent temporal units help once graph communication is absent.

For each individual contrast, use the existing 10,000 paired-article bootstrap
draws with seed 31415 and report its 95% interval. These intervals condition
on the two fitted checkpoints; they do not estimate uncertainty from training
seeds, graphs or datasets. Contrasts share references and articles. Do not pool
repeated seeds as independent documents, manufacture a confidence interval
for their descriptive mean, or claim equivalence from an interval crossing
zero. No equivalence margin is declared.

Publish positive, null and negative results. A full-model advantage would
support that component under this exact parameterization and budget, with the
reported trainable-parameter differences. A fixed-dynamics comparison cannot
establish an anatomical advantage without retraining on null graphs. The
completed topology study demonstrated no advantage of the retained measured
wiring over its declared rewired controls. These new controls do not reverse
or replace that finding. No conclusion licenses claims about an intact fly
brain, biological learning, language understanding or connectomes generally.

After this study, resume the saved BabyLM comparison and the broader data,
domain and behavioral research program. Keep their splits and declarations
separate from this small-corpus mechanism test.

## Commands

```powershell
python -X utf8 -m unittest discover -s tests -p "test_language_core*.py"
python -X utf8 -m flm.language_core_study prepare
python -X utf8 -m flm.language_core_study train
python -X utf8 -m flm.language_core_test freeze
python -X utf8 -m flm.language_core_test score
```

The training command resumes verified committed checkpoints. The scoring
command refuses incomplete runs; it does not train missing conditions.
