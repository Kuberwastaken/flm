# Train-only timing pilot

The [pilot command](../flm/scan_pilot.py) is prepared and fixture-tested. No
full-size pilot has run and no downstream benchmark budget has been selected.
The [preflight record](../reports/scan-runtime/pilot-preflight.json) distinguishes
software tests from an observed timing result.

Run it only after the registered BabyLM and chosen neuron-selection comparisons
finish, and after checking their live process handles have exited. The pilot
command checks the earlier language-computation and BabyLM completion markers;
the [study coordinator](SCAN-STUDY-COORDINATOR.md) also verifies the completed
selection comparison. An OS-held lease excludes another pilot writer.
Completion files alone cannot establish that no other training process is active.

The pilot includes all three SCAN splits, all three architectures, and both
original and WikiText initializations at seed 42: eighteen conditions. It loads
only official training records. Each condition starts from a verified source,
then updates a disposable copy for three warmup and twelve measured updates.
All use batch 16, four CPU threads, the complete action/EOS loss and sampling
seed 42. The short pilot uses AdamW, learning rate 0.002 with three-update warmup
and cosine decay to 0.0002 over fifteen updates, weight decay 0.01 and norm clip
1.0. These are fixed timing settings, not an optimizer search or the later
benchmark's chosen schedule.

Timing includes sampling, tokenization/padding, forward/backward computation,
clipping and the optimizer step. Source preparation, copying, integrity checks,
checkpoint I/O and measurement bookkeeping are excluded. The report retains
every update duration, separates warmup exposure, and hashes the sampled row
stream. Short-run throughput cannot establish long-run speed, thermal behavior,
peak memory or a hardware-independent advantage. Account for omitted overhead
when declaring the eventual common update budget across all 36 benchmark fits.

Pilot weights and optimizer states are never saved or reused. Original model
weights and the caller's RNG state remain unchanged. Each completed timing case
is recorded in a dated attempt directory; a failure retains earlier cases and
its diagnostic. A completed pilot report cannot be overwritten. Failed attempts
are not silently relabeled successful or omitted from the local record.

```sh
python -m unittest discover -s tests -p test_scan_pilot.py -v
# Only after the priority comparisons finish and their trainers have exited:
python -m flm.scan_pilot
```

The five tests use synthetic cases and three-update tiny neural fixtures. They
verify exact row/exposure accounting, warmup exclusion, source/RNG preservation,
priority refusal and retention of failed attempts. Their durations are not
measurements of the actual benchmark models.
