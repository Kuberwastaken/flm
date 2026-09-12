# Mac concurrency qualification: retain serial execution

On 12 September 2026, the user requested increased Mac training throughput. We compared one, two, three and four independent spawned workers on the Apple M5 Pro, keeping the existing native model, float32 computation and four CPU threads per fit. This was an execution probe, not a language or anatomical result.

The [prospective scheduling amendment](SELECTION-MAC-PARALLELISM.md) required at least a 15% aggregate throughput improvement and exact final model and optimizer tensor hashes before changing the worker count. It preferred the smallest worker count within 10% of the best qualifying throughput. These rules were written and committed before timing.

| Workers | Aggregate updates/second | Relative to serial | Exact final tensors |
|---|---:|---:|---|
| 1 | 19.191 | 1.000 | Reference |
| 2 | 19.029 | 0.992 | Match |
| 3 | 19.817 | 1.033 | Match |
| 4 | 21.882 | 1.140 | Match |

**Decision: retain one worker.** Four workers achieved a 14.03% aggregate gain, below the declared threshold. We did not move the threshold after seeing the result. The prepared parallel scheduler was not adopted. The 20 completed native fits were preserved byte-for-byte, the serial adapter resumed with the same native identity, and the interrupted fit can recover its latest committed checkpoint. Validation and held-out gates remain unchanged.

Each worker used the fixed 540-neuron right-seeded candidate, training seed 42, ten warmup updates and forty timed updates on deterministic synthetic token inputs. Timed intervals excluded process startup and warmup; the aggregate denominator spans the first worker start through the last worker finish. The [raw probe](../reports/selection-language/mac-v1/parallel-probe-v1.json) preserves all timings, model/parameter/optimizer hashes and source bindings. The [driver](../scripts/selection_mac_parallel.py) implements the declared choice.

These brief, single-round measurements do not establish sustained throughput across all graphs, full-trajectory numerical equality, energy efficiency, or that parallel training cannot help other configurations. The Mac was connected to AC power when checked after qualification, with no reported thermal or performance warning. We did not change persistent power settings, per-fit numerical settings, exposure budgets, data, graph controls or interpretation thresholds.

The [handoff](../reports/selection-language/mac-v1/parallel-handoff.json) retains pre-pause completion hashes; the [decision](../reports/selection-language/mac-v1/parallel-decision-v1.json) binds the probe, handoff and unchanged native identity. Operations and current remote paths are documented in [Mac training operations](MAC-TRAINING-OPERATIONS.md).
