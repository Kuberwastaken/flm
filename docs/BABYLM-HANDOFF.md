# From the fixed BabyLM queue to selection costs

**Completed 12 September 2026:** the corrected v2 sequence finished all twelve
held-out evaluations, all 288 fixed continuations and the 64-original timing
pilot. See [complete findings](BABYLM-FINDINGS.md) and the
[evidence audit](../reports/babylm/completed-evidence-verification.json).
The first attempt and its recovery below remain part of the record.

Kuber Mehta · 12 September 2026. The
[local supervisor](../scripts/continue_babylm_research.py) is an operational
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

The [fourteen tests](../tests/test_babylm_handoff.py) exercise process identity,
PID reuse, access denial, an observed child outliving its parent, competing work,
exclusive supervisors, incomplete training, source/package drift, existing
outputs, command failure and shortened result inventories. The complete
sequence test substitutes artificial completion/output records and a fake stage
runner. **It executes no real benchmark or timing command.** The
[verification record](../reports/babylm/handoff-preparation.json) distinguishes
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
The [declared BabyLM evaluation](BABYLM-EVALUATION.md),
[selection cost plan](SELECTION-TIMING-PILOT.md), and
[complete-group coordinator](SELECTION-LANGUAGE-COORDINATOR.md) remain the
authoritative scientific definitions.

## Auditing one finished fit

The [completion audit command](../scripts/audit_babylm_completion.py) replaces
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
The [four command tests](../tests/test_babylm_completion_command.py) cover
incomplete fits, overwrite refusal, protocol drift and identity/inventory changes.
The [verification record](../reports/babylm/completion-command-preparation.json)
also distinguishes those artificial fixtures from an
[actual completed-fit audit](../reports/babylm/completion-command-transformer-100m-s42.json)
that reproduces the previous 100M transformer seed-42 record exactly.

The [100M FLM seed-43 completion audit](../reports/babylm/completion-flm-100m-s43.json)
checks the tenth completed fit: all 24 validation records and the selected/final
checkpoints pass. Update 12,000 is selected at 1.9318626804 validation bits per
byte after 18,432,000 presented tokens. This is checkpoint-selection evidence;
the complete-study held-out evaluation remains pending.

The [100M GRU seed-43 completion audit](../reports/babylm/completion-gru-100m-s43.json)
checks the eleventh completed fit with the same 24-record and checkpoint checks.
Update 12,000 is selected at 1.8563063670 validation bits per byte after
18,432,000 presented tokens. The final transformer fit and complete-study
held-out evaluation remain pending.

## All fits complete; held-out scoring started

The [final 100M transformer seed-43 audit](../reports/babylm/completion-transformer-100m-s43.json)
checks the twelfth completed fit. Update 12,000 is selected at 1.8462591591
validation bits per byte after 18,432,000 presented tokens.

The supervisor observed the trainer and all observed children exit and started
held-out scoring at 2026-09-11 21:54:10 UTC. The [twelve-run selection](../reports/babylm/selection.json)
was frozen before scoring. A subsequent [verification](../reports/babylm/selection-verification.json)
reproduces it exactly with the frozen verifier, restoring selected/final
checkpoints and checking all 288 validation records. It hashes and maps all
prepared caches, including test data, for identity and boundary checks; it does
not run model inference or calculate test likelihoods. The verification record
corrects the original overly broad claim that it did not access test tokens.

This is a dated operational milestone, not a completed held-out comparison.
Fixed continuations and selection timing remain later stages of the same
supervised queue; the earlier arming and single-fit records above are historical.

## Evaluation recovery

The first held-out attempt stopped on 11 September at 22:25:55 UTC after scoring
all 3,187 blocks for its first model. Result assembly supplied `checkpoint_step`
both through the frozen selection and as a separate keyword, causing a
`TypeError`. No completed model result or study summary was written. Training,
checkpoint selection and numerical scoring were not changed by the repair.

The duplicate keyword was removed. A regression exercises the actual main
result-writing path for twelve artificial selections containing this field,
with toy scored blocks; it writes and checks all results and the summary. The
same test reproduces the error with the original statement. Together with
selection, cache and supervisor checks, 26 tests pass. These are software
checks, not additional scientific results.

The stopped supervisor and evaluator were confirmed absent. All 400 files from
the failed cache (399 batch records and one identity) are preserved locally at
`runs/babylm-evaluation-failed-20260911-v1`, with every byte hash checked before
and after moving. They cover 18,177,107 tokens and 51,722,871 bytes. The old
supervisor directory and log remain intact. Because evaluator source identity
changed, the corrected attempt scores afresh instead of relabeling old batches.
The frozen twelve-model selection remains byte-identical.

An explicit completed-training mode permits a new supervised attempt without
inventing a live trainer PID. It verifies all twelve completed fits, takes the
same exclusive lease, binds current source/package identities, waits for known
jobs to exit, and preserves the existing-output and full-inventory gates.
Missing-fit and busy-job tests verify that it cannot bypass these prerequisites.
It does not automatically retry a failure.

```powershell
python -u -X utf8 scripts/continue_babylm_research.py --completed-training --output work/babylm-post-training-v2
```

The [recovery record](../reports/babylm/evaluation-recovery.json) retains the
failed attempt, cache preservation, software tests and corrected source hashes.
The corrected queue retains the original order: complete held-out evaluation,
fixed continuations, then selection timing. Complete-study results remain pending.
