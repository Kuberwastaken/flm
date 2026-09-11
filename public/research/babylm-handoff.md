# From the fixed BabyLM queue to selection costs

Kuber Mehta · 12 September 2026. The
[local supervisor](babylm-handoff/continue_babylm_research.py) is an operational
handoff for the existing research plan. It waits for a specifically identified
live BabyLM suite and its observed descendants, then runs these existing commands
serially:

1. `python -m flm.babylm_test`: restore and audit every selected/final checkpoint,
   freeze all twelve selections, and perform the declared complete held-out
   likelihood evaluation.
2. `python -m flm.babylm_samples`: produce all 288 fixed-prompt continuations,
   preserving the declared sampling settings and failures.
3. `python -m flm.selection_pilot`: measure the complete disposable 64-original
   graph cost inventory, after the earlier commands exit.

The supervisor changes none of those scientific commands or their protocols.
It chooses no selection groups, optimizer schedule or training budget, and
does not launch selection-language fits. The measured costs still need review
before choosing complete groups and freezing that experiment. It also does not
commit, push, deploy, generate figures or interpret the resulting scores.

## A live process is the prerequisite

Arming requires the actual suite PID, its creation time, the exact
`flm.babylm_suite` module command and this repository's working directory.
The wait follows the observed child processes too: the parent exiting alone
does not establish that its last training child has stopped. PID reuse counts
as the original process having exited. Inspection failures and access denials
for bound processes propagate; they are not converted into an exit event.

An operating-system lease permits one such supervisor per repository. Before
each stage, it waits for known competing project training, evaluation, timing
or unittest module commands. This is not a guarantee that the machine is idle:
unrelated programs and unrecognized entry points can still consume resources.
Inspect the live handles and avoid other computational work during the cost
pilot. A lock file or a previous status message alone is not a live-process check.

After the trainer exits, a preliminary gate requires twelve 12,000-update
completion records and matching selected-checkpoint checksums. The complete
tensor, validation-history and corpus audits remain inside the existing
BabyLM evaluator, before it opens test-token payloads.

The handoff binds the relevant source/protocol/prompt bytes and installed
NumPy, Torch, SciPy, tokenizers and psutil versions at arming. It checks them
before and after each command. Ongoing changes to unrelated notes or software
do not justify changing these frozen numerical sources mid-study.

## Failure and recovery

The handoff requires a fresh output directory. It records its identity,
timestamped events, stage child PIDs/creation times, and a separate log for each
command. Successful commands must also produce the complete expected output
inventories: twelve scored models, twelve models × twenty-four passages, and
sixty-four timed graph selections. These inventory checks supplement the
commands' own scientific verification; they are not an independent reanalysis.

A failed command stops the sequence. An already existing stage output also
stops it, preserving that record for inspection rather than rerunning it.
There is no automatic retry or overwrite of completed costs. Existing model
and evaluation caches retain their native recovery behavior when an operator
later resumes the appropriate command.

If the supervisor itself is interrupted, inspect its recorded stage child
before restarting anything. Stopping the supervisor does not prove a child
exited. Do not start a second evaluator or timing run based only on a missing
supervisor handle. The final handoff record is written only after all three
commands exit successfully and their inventories pass.

## Reproduction and current scope

The [fourteen tests](babylm-handoff/test_babylm_handoff.py) exercise process identity,
PID reuse, access denial, an observed child outliving its parent, competing work,
exclusive supervisors, incomplete training, source/package drift, existing
outputs, command failure and shortened result inventories. The complete
sequence test substitutes artificial completion/output records and a fake stage
runner. **It executes no real benchmark or timing command.** The
[verification record](babylm-handoff/preparation.json) distinguishes
these tests from the observed live wait and the still-closed training gate.

```powershell
# Optional process-supervision dependency, already present on the working host:
python -m pip install -r requirements-operations.txt
python -m unittest discover -s tests -p test_babylm_handoff.py -v
# Replace 12345 with the verified live suite PID; use a fresh output directory:
python -u -X utf8 scripts/continue_babylm_research.py `
  --trainer-pid 12345 --output work/babylm-post-training-new
```

The launched handoff is not a new result or a revised research priority.
The [declared BabyLM evaluation](babylm-evaluation.md),
[selection cost plan](selection-timing-pilot.md), and
[complete-group coordinator](selection-language-coordinator.md) remain the
authoritative scientific definitions.

## Auditing one finished fit

The [completion audit command](babylm-handoff/audit_babylm_completion.py) replaces
per-condition one-off scripts with explicit scale, architecture and seed options.
It requires a completed fit and all 24 validation records before loading model
software or validation data. It checks the fixed protocol independently of other
runs, the study inventory, the eight numerical sources against the training
commit, and the installed Torch version. The existing payload verifier restores
selected/final checkpoints and checks every recorded validation observation.
Input and source hashes must remain unchanged through verification.

```powershell
python scripts/audit_babylm_completion.py --scale 100m --seed 42 --variant transformer `
  --output work/another-completion-audit.json
```

Use a fresh output path and run the auditor from the target repository checkout.
It performs no model forward pass or test-cache access and does not freeze the
twelve-run selection. The supervisor still applies its own complete-study gate.
The [four command tests](babylm-handoff/test_babylm_completion_command.py) cover
incomplete fits, overwrite refusal, protocol drift and identity/inventory changes.
The [verification record](babylm-handoff/completion-preparation.json)
also distinguishes those artificial fixtures from an
[actual completed-fit audit](babylm-handoff/completion-transformer-100m-s42.json)
that reproduces the previous 100M transformer seed-42 record exactly.

The [100M FLM seed-43 completion audit](babylm-handoff/completion-flm-100m-s43.json)
checks the tenth completed fit: all 24 validation records and the selected/final
checkpoints pass. Update 12,000 is selected at 1.9318626804 validation bits per
byte after 18,432,000 presented tokens. This is checkpoint-selection evidence;
the complete-study held-out evaluation remains pending.

The [100M GRU seed-43 completion audit](babylm-handoff/completion-gru-100m-s43.json)
checks the eleventh completed fit with the same 24-record and checkpoint checks.
Update 12,000 is selected at 1.8563063670 validation bits per byte after
18,432,000 presented tokens. The final transformer fit and complete-study
held-out evaluation remain pending.
