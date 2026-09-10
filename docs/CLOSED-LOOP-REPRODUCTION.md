# Reproduce the physical feedback assay

Kuber Mehta · FLM / ChatFLM

The records archive contains all 27 physical conditions, the exact repeat,
the four fixed sensory-choice checkpoints, their numerical parity fixtures,
source files, the declared protocol and component licenses. It does not require
access to the private repository. These checkpoints take four sensory inputs
and predict two actions; they are separate from the language models.

## Audit existing observations

Extract `closed-loop-records.zip` to a new directory and open a terminal there.
Use Python with NumPy installed. The recorded simulator environment used Python
3.12.14 and NumPy 2.5.3; `requirements-embodied-lock.txt` lists the full physical
environment. The audit itself needs NumPy and the included Python files, without
FlyGym, MuJoCo, PyTorch, a GPU or a display:

```sh
python scripts/audit_feedback_release.py
```

The audit first verifies every manifest payload and the four-model parity
fixture. It then independently derives the sampled sensory cue from the
recorded body pose, replays every neural frame, verifies the delayed query
decision and reconstructs the actual motor-command timeline. It recomputes
angular error, distance, phase-specific progress and the other metrics, then
checks the repeated physical and neural arrays for exact equality.

The complete cohort contains 54,027 physical observations, 10,800 control frames
and 972 delayed decisions. The additional repeat contains 2,001 observations,
400 frames and 36 decisions. The three scripted cases have sensory and command
records but no invented neural state or probabilities. State/logit parity uses
absolute tolerance 2e-6 and relative tolerance 2e-5 together. A different NumPy
runtime can produce small arithmetic differences; a failed tolerance check must
be investigated rather than hidden by relabeling the record.

The manifest establishes file identity, not a third-party signature. An audit
of recorded data is distinct from rerunning MuJoCo physics. The archive also
includes the original predeclared video for the eligibility/live/switch case.
Two simulated seconds play in eight seconds. The target is a virtual coordinate;
the body camera does not depict it as a visible object.

## Run fresh physics

Use the pinned physical environment and an **empty, separate experiment
directory**. Copy these paths from the archive, preserving their layout:

- `experiments/embodiment/` for the simulator, controller and verifier;
- `data/controllers/choice-v1/` for all four models and parity fixtures;
- `docs/CLOSED-LOOP-PROTOCOL.md` and `requirements-embodied-lock.txt`;
- `scripts/audit_feedback_release.py` for a subsequent numerical audit.

Do not copy the existing `runs/` directory or its root manifest into a fresh
experiment. From the new directory run:

```sh
python experiments/embodiment/closed_loop.py
python scripts/audit_feedback_release.py
```

The runner creates an identity from its sources, inputs and current environment,
runs all 27 cases and repeats the predeclared case. On an interrupted run, use
the same command after confirming that its previous process stopped. Completed
trials resume only when their identities and artifact hashes agree. An OS-held
lock prevents two simultaneous physical writers. A failed trial is recorded
explicitly and requires investigation; the runner will not silently replace it.
Do not overwrite the downloaded observations with a new run.

Exact physical equality is established for the repeated seed in the recorded
environment. Differences in platform, simulation libraries or floating-point
behavior can change a fresh trajectory. Preserve its new identity and report
those differences. One repeated seed is not a population uncertainty estimate.

## Read the results

The website's summary JSON and CSV report all conditions. The complete records
include the 1 ms body traces and 5 ms control frames. Compare live and frozen
pose at the same model and scenario. Negative live-minus-frozen mean angular
error favors live feedback. Target-switch progress is computed separately
against each phase's fixed target, avoiding an artificial distance jump when
the target changes.

The scripted reference receives the same cue, 50 ms delay, straight-ahead
deadband and gait interface. Identical generalized-position arrays are reported
as identical physical paths even if their neural states or logits differ.
Neither a successful learned choice nor a shared path demonstrates that learned
recurrent dynamics, anatomical wiring, language knowledge or learned gait were
necessary. The hypothesis about anatomical wiring is tested in the separate
matched language topology study.

See the included [declared protocol](CLOSED-LOOP-PROTOCOL.md) and
[original learning protocol](LOCAL-LEARNING-PROTOCOL.md). The physical body,
hybrid gait and contact feedback follow the official
[FlyGym turning tutorial](https://neuromechfly.org/tutorials/4d_turning_controller/).
