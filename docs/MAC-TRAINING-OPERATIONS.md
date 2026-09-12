# Operating the active Mac training queue

The active experiment is the complete native cohort declared in [SELECTION-MAC-EXECUTION.md](SELECTION-MAC-EXECUTION.md). The Windows attempt is stopped and retained. Do not infer failure from the absence of the old Windows PIDs or restart its command.

For this workspace, private connection details are in the ignored `work/mac-runtime-connection.json`: tailnet host, SSH user, remote project root, interpreter, log and supervisor receipt. Use those values rather than guessing a username or copying account details into public documentation. Connect through the existing Tailscale authorization; the local known-hosts file contains host keys supplied by the tailnet. No credentials belong in the repository.

The initial remote `work/mac-selection-supervisor-v1.json` records the first launch. After the bounded concurrency probe, `work/mac-selection-resume-supervisor-v1.json` records the active serial restart. Rediscover and verify the actual process command and creation/elapsed information before using its PID; process IDs can be reused. `work/mac-selection-resume-v1.log` is the active combined train/select/test log; the initial log is retained. A process-bound `caffeinate` instance keeps the machine awake for this job without changing persistent power settings. Keep the Mac powered and reachable.

The source-bound adapter and its amendment, runtime profile, native identity and original numerical sources must not change while the queue runs. Normal startup and stage boundaries recheck the complete native context. An interrupted fit can resume only within this same native runtime, retaining its optimizer, random state and exposure records. The OS-held study lease prevents concurrent writers; the existence of a lock file alone does not mean a worker is active.

The remote study paths are:

| Artifact | Path relative to the remote project |
|---|---|
| Training records | `runs/selection-language-mac-v1/` |
| Frozen native identity and profile | `reports/selection-language/mac-v1/` |
| Complete checkpoint selection | `reports/selection-language/mac-v1/study-selection.json` |
| Held-out summary and per-fit reports | `reports/selection-language/mac-v1/test-summary.json` and `test/` |
| Held-out identity and scoring batches | `runs/selection-language-mac-evaluation-v1/` |

The initial transfer verified 150 training/context files. The fixed held-out cache was transported separately as opaque bytes for the later scorer; no token decoding or model scoring was performed during transport. The training/initialization code opens only held-out metadata, and the original full-inventory selection gate precedes held-out inference.

After completion, run the following **on the Mac**, where the complete results exist:

```sh
.venv/bin/python scripts/selection_decision_report.py --mac-study
# Only for a complete valid record; preserve pending snapshots under other names:
.venv/bin/python scripts/selection_decision_report.py --mac-study --output reports/selection-language/mac-v1/allocation-result-v1.json
```

The Windows mirror initially contains only setup and handoff evidence. Its local reporter will remain pending until the complete native result/selection records and evaluation identity are copied back. Never mistake an unsynchronized mirror for remote training failure. Transfer complete, hash-bound records without overwriting the historical Windows study. The reporting consumer needs every native per-fit result; reproducing inference additionally needs the selected checkpoints, original inputs and evaluation caches.

If the supervisor has exited, inspect its final log and artifacts before deciding whether it succeeded or needs recovery. Retain errors and partial records. Do not replace a failed condition, lower the allocation threshold, or declare the overall FLM goal complete because this queue ends. Follow the [current plan](PLAN.md) and [decision guide](SELECTION-DECISION-REPORT.md).

## Concurrency qualification outcome

The user requested higher Mac throughput on 12 September. The [bounded concurrency qualification](MAC-CONCURRENCY-FINDINGS.md) compared one through four isolated workers at the original four threads per fit. The best short-probe aggregate gain was 14.02%, below the declared 15% adoption threshold. The active queue therefore remains the original serial adapter. Twenty completed native fits were preserved byte-for-byte, and the interrupted fit resumes only from its last committed native checkpoint. No parallel execution record was created, and no new native cohort was started.

Do not automatically rerun the probe, adopt four workers, or launch the prepared parallel scheduler. Preserve its measured result and the [decision record](../reports/selection-language/mac-v1/parallel-decision-v1.json). The original numerical sources, native adapter, identity, runtime profile and execution amendment remain frozen.

## Scheduled monitoring

The existing 30-minute training heartbeat remains active. Its saved legacy prompt already requires reading AGENTS.md and this plan first; those current repository instructions override its old Windows launch details and broader obsolete queue. The app acknowledged another prompt update on 12 September, but the saved automation file still contained the previous text when inspected. Do not claim the new prompt persisted, recreate duplicate schedules, or modify automation configuration by hand.

Use `scripts/mac_training_status.py` through the authenticated remote Python interpreter, passing the current supervisor receipt from the ignored connection record. The helper reads declaration/completion/checkpoint file metadata and ps/pmset telemetry without loading model tensors, training text or partial held-out scores. Keep the previous and latest snapshots in ignored `work/mac-training-monitor-previous.json` and `work/mac-training-monitor-latest.json`. A snapshot is telemetry, not a scientific-completion audit.

Compare completed-fit count, active committed checkpoint and timestamps across checks. In the training phase, no newer committed checkpoint or completed fit across a full check interval warrants investigation; confirm process identity, file changes and logs before calling it a stall or attempting recovery. A stage change is progress: validation and held-out work require their own artifact/log checks, not a continuing training-checkpoint counter. Assess slowdowns across several comparable fits and distinguish graph differences, power changes and thermal warnings from a process failure. Never restart because CPU utilization is low.

Report both process CPU percentage and its approximate share of total logical CPU capacity. At the initial monitoring check the Mac had 15 logical CPUs and the trainer used about 1.5-1.6 cores, roughly 10-11 percent of the whole machine. Recent completions continued despite that low share. The earlier four-worker probe gained only 14.02 percent aggregate throughput, below its declared adoption criterion; this does not identify the underlying hardware/software bottleneck or prove that another execution approach could never help. Preserve the current study and do not launch another optimization campaign automatically.

Stay quiet for routine progress. Notify on meaningful stage/completion milestones, suspected stalls, sustained material throughput regressions, failures, or required action. After the complete native result becomes available, perform the existing record-bound audit and continuation decision. Keep all thresholds and the broader paused-goal constraints unchanged.
