# Measuring the cost of alternative language-learning rules

The [training-only input adapter](LANGUAGE-LEARNING-INPUTS.md) and
[resumable runner](LANGUAGE-LEARNING-RUNNER.md) now have a disposable cost pilot.
It uses the actual gradient/update paths for BPTT, fixed recurrent dynamics,
eligibility traces and the no-history control. **The official timing pilot has
not run, and no language-learning budget has been selected.**

## What is ready

The [full-window preflight](../reports/language-eligibility/pilot-preflight.json)
verifies all eight conditions: four learning rules crossed with initialization
and sampling seeds 42 and 43. It loads the pinned original 1,024-neuron,
76,130-edge subset and the verified BabyLM 2026 English `train-10m` cache, using
the shared 4,096-entry train-fitted tokenizer. All 3,351 training blocks are
eligible for the declared 96-token windows. Initial parameter tensors, the
ordered training inventory and sampled windows match across rules within each
seed. Different seeds deliberately use different initializations and windows.

This preflight executes declarations and sampling, **zero gradient updates**.
It opens no validation or test payloads and records no timings or language
scores. The earlier tiny update checks remain separate evidence. A successful
sampling check does not establish memory feasibility or runtime at the full
batch and sequence dimensions.

The declared measured segment, after three warmup updates, has this exact
sampled exposure for every learning rule:

| Seed | Input tokens | Scored targets | Input UTF-8 bytes | Scored UTF-8 bytes |
|---|---:|---:|---:|---:|
| 42 | 18,432 | 15,360 | 53,699 | 44,921 |
| 43 | 18,432 | 15,360 | 54,697 | 45,573 |

The mask excludes the first 16 positions and boundary-token targets. None of
these particular measured windows contains a boundary target after warmup;
boundary masking is independently exercised by the artificial unit fixtures.
This does not change the earlier language trainer's mask or make its results
a matched BPTT reference for the new learning-rule study.

## What the pilot will measure

[language_learning_pilot.py](../flm/language_learning_pilot.py) visits all eight
conditions in one fixed PCG64-shuffled order, with order seed 617. Each condition
starts from fresh random parameters, makes three warmup updates, then measures
twelve updates at batch 16, sequence length 96 and four CPU threads. It uses
float32 through the official input adapter, 96-dimensional embeddings, 128 pools
and the tied readout. The fixed-core condition trains 422,496 parameters; the
other three train 600,003. This fixed-core condition freezes the input projection
and bias as well as edge magnitudes, time constants and gain. It differs from
the earlier fixed-dynamics control, which learned its input projection. Frozen
parameters remain part of the forward computation.

The pilot calls the existing `language_learning_train.update` implementation;
it does not replace eligibility learning with a cheaper forward-only proxy.
All conditions use AdamW at a constant pilot learning rate of 0.002, weight decay
0.01 and gradient clipping at 1.0. This constant pilot rate measures update cost;
it does not select the future study's learning-rate schedule or convergence.

The timed region includes sampling and the complete shared update function:
parameter-inventory checks, masking, gradient computation, finite checks,
clipping and AdamW. It excludes loading, declaration verification, initialization,
exposure/hash bookkeeping, the extra optimizer-state audit, checkpoint I/O and
validation. The report retains every update time, measured token and byte totals,
median update time and aggregate throughput. It verifies matching initialization
and exposure within each seed before accepting a condition into the report.

Eligibility tensor accounting is recorded separately. A zero trace count for
BPTT or fixed-core means that no explicit eligibility trace was allocated; it
does not mean zero activation memory. No peak-memory or energy measurement is
implemented here. One shuffled pass cannot establish long-run thermal stability
or a reliable total training wall time.

The pilot copies supplied models, checks that originals remain unchanged, and
checks fixed recurrent parameters after updates. It discards every optimizer
and trained model. It saves no loss values, predictions or raw text. Completed
conditions from a failed attempt remain available with a failure record; a new
attempt starts fresh rather than combining timings from different sessions.
An already completed report cannot be overwritten by this command.

## Verification and launch

Six tests exercise actual updates for all eight tiny fixture conditions,
independent masked token/byte arithmetic, both sample digests, fixed-core and
source preservation, restored global RNG/thread settings, invalid settings,
update failures, prerequisite checkpoint hashes, failed-attempt retention,
complete inventories, overwrite refusal and mismatched exposure rejection.

```powershell
python -m unittest discover -s tests -p test_language_learning_pilot.py -v
python -m scripts.language_learning_pilot_preflight --output reports/language-eligibility/another-dated-pilot-preflight.json
# After priority training has completed and actual process handles have exited:
python -m flm.language_learning_pilot
```

The official command checks all twelve BabyLM completion records and selected
checkpoint hashes before opening its training inputs. The current gate correctly
refuses because that queue is unfinished. Files are not process locks: inspect
the active training and timing handles before launching. An OS-held directory
lock excludes concurrent writers of this pilot. Its completed report will be
`reports/language-eligibility/timing-pilot.json`; no such result exists yet.

After measuring costs, freeze a common exposure budget, learning-rate schedule,
validation-selection rule and all-condition test gate, then train fresh matched
models. This varies the learning rule on the original subset. The separate
selection-method experiment, existing BabyLM comparison and completed sensory
and physical studies retain their own protocols.
