# Operating the active Mac training queue

The active experiment is the complete native cohort declared in [SELECTION-MAC-EXECUTION.md](SELECTION-MAC-EXECUTION.md). The Windows attempt is stopped and retained. Do not infer failure from the absence of the old Windows PIDs or restart its command.

For this workspace, private connection details are in the ignored `work/mac-runtime-connection.json`: tailnet host, SSH user, remote project root, interpreter, log and supervisor receipt. Use those values rather than guessing a username or copying account details into public documentation. Connect through the existing Tailscale authorization; the local known-hosts file contains host keys supplied by the tailnet. No credentials belong in the repository.

The remote `work/mac-selection-supervisor-v1.json` records the launch. Rediscover and verify the actual process command and creation/elapsed information before using its PID; process IDs can be reused. `work/mac-selection-v1.log` is the combined train/select/test log. A process-bound `caffeinate` instance keeps the machine awake for this job without changing persistent power settings. Keep the Mac powered and reachable.

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
