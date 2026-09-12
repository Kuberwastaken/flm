# Mac concurrency amendment

Kuber Mehta · 12 September 2026. Authorized by the request to ramp up the Mac. This changes scheduling of independent fits within the existing native cohort, not its identity, initialization, numerical source, per-fit settings, training budget, validation policy or anatomical continuation criterion.

The initial serial native attempt is retained, including all completed fits and recoverable checkpoints. Stop its verified supervisor before running the concurrency probe or new coordinator. Unlike the earlier Windows-to-Mac migration, this change uses the same native runtime and can resume its exact recorded optimizer, RNG and exposure state. Do not restart the cohort from scratch or combine Windows results with it.

## Bounded qualification

Measure one, two, three and four spawned workers on the same declared 540-neuron native model. Each uses four CPU threads, ten warmup and forty timed synthetic-token updates with the original optimizer/update function and fixed inputs. There is no official writer or corpus scoring during this probe. Compare complete final model and optimizer tensor hashes with the serial probe. Only configurations with exact agreement qualify.

Choose the fewest workers within 10% of the best qualified aggregate update throughput. Require at least 15% improvement over one worker to increase concurrency. These are operational allocation thresholds selected before the timings, not language-result criteria. The short probe does not prove numerical equality for every graph or full trajectory, nor guarantee sustained throughput under a different battery/thermal load. No model score influences the scheduling choice.

## Ownership and recovery

The new parent coordinator holds the existing OS study writer lease while its spawned workers run. Each of the 128 original condition indices is submitted exactly once; processes own separate model, optimizer, RNG and run-directory state. The original per-run writer lease remains in force. Completed fits are restored and audited without additional updates, and their completion hash must remain unchanged. Interrupted fits resume only committed checkpoints. Every fit still receives exactly 3,000 updates and the original checkpoint schedule.

The [parallel driver](../scripts/selection_mac_parallel.py) is additional execution source. Its hash, this document, the qualification report, worker count and full condition inventory are bound in a separate `parallel-execution-v1.json` alongside the native identity. The existing runtime adapter and its amendment stay byte-identical. Preserve the supplementary scheduling records with the eventual study results.

An error prevents validation/test launch and retains all recoverable files; other interrupted workers may lose only uncommitted work. All 128 worker results must be present and verified, and all worker processes must exit, before the existing serial full-inventory validation and held-out routines run. No partial-study evaluation, dropped condition, reduced budget or changed scientific threshold is permitted.

CPU utilization is not a scientific outcome. Monitor sustained fit throughput and battery/thermal conditions. Further scheduling changes must be recorded separately instead of silently overwriting this execution record. The queue remains part of the same bounded anatomical-prior decision.
