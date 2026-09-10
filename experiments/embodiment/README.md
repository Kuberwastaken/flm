# Embodied experiments

Use Python 3.12 or newer in an isolated environment. Install the exact environment
from `requirements-embodied-lock.txt` with `uv pip install --python PATH -r FILE`.
The WikiText trainer retains its original Python 3.10/PyTorch environment.

Run `python experiments/embodiment/calibrate.py --video` first. This uses FlyGym
2.1.0 and MuJoCo physics, with the upstream hybrid walking controller and its
preprogrammed leg trajectories. It tests symmetric and asymmetric descending
commands for one simulated second from the same seed, and repeats the symmetric
trial to verify deterministic physical replay. It records positions, orientations,
contacts and full generalized positions. A video is optional and is included in
the reported wall time when enabled.

This first calibration has no FLM connection and no learning. Its purpose is to
establish a reproducible plant and action interface before comparing neural
controllers. Distances are millimeters and time is seconds, following FlyGym's
model conventions. Orientation and height are diagnostic measurements; neither
alone is presented as a validated fall detector.

Controller and model source: [FlyGym turning tutorial](https://neuromechfly.org/tutorials/4d_turning_controller/)
and the installed FlyGym 2.1.0 distribution. Preserve its Apache-2.0 source license
and asset-specific notices when redistributing upstream components.

## Learned-choice replay

Complete `python -m flm.behavior_study` in the main PyTorch environment first.
In the isolated physical environment, run:

```sh
python experiments/embodiment/learned_choice.py --video
```

This runs all five learning rules, checkpoints 0/300/600/900, and both fixed cues
from seed 17: forty independently simulated one-second trials. A neural argmax
choice selects one of two calibrated descending commands. Every case records
its input, probabilities, full neural state, checkpoint identity and generalized
physical positions. Completed trials resume only when identities match. The
video case is declared in advance: supervised eligibility, update 300, cue 0.

Back in the research environment, `python scripts/physical_choice_report.py`
verifies all forty trajectories and publishes the video, figures, case records
and CSV archive. It requires FFmpeg on PATH and moves MP4 metadata to the start
for streaming without re-encoding video frames; both file hashes are recorded.
All headings follow their chosen command; erroneous neural
choices remain in the record. Identical commands produce identical trajectories
from the shared physical initialization. No online neural sensory feedback,
learned balance, learned gait or language-to-motor transfer is tested here.

## Online pose feedback

The next [declared assay](../../docs/CLOSED-LOOP-PROTOCOL.md) uses current
simulated position and heading to form a new sensory cue every 55 ms. The
trained network processes eleven frames before issuing each high-level choice;
it resets between cycles to match its original training. Frozen-pose controls
keep the same changing waypoint schedule, and a scripted reference exposes the
contribution of the engineered feedback/deadband/gait wrapper.

The shared initialization and final BPTT, fixed-core and eligibility checkpoints
have a NumPy inference package under `data/controllers/choice-v1`. The exporter
in the original PyTorch environment verifies all three initial states agree and
creates 128-frame parity fixtures for each of the four models. Both environments
pass state/logit checks at absolute 2e-6 and relative 2e-5, with identical argmax.
To reproduce the export when the original cue checkpoints are available:

```sh
python scripts/export_choice_runtime.py
```

The committed package suffices to run physics in the isolated environment:

```sh
python experiments/embodiment/closed_loop.py
```

This runs all 27 conditions plus an exact repeat of the predeclared video case.
It owns an OS-held lease for the output directory, resumes only verified
completed cases, limits numerical thread pools to one and lowers its own
Windows process priority. It does not change the language trainer's priority,
threads or frozen sources. Never start another writer while it is live.
`--case eligibility-live-switch` runs only that declared case; full publication
still requires every condition and the repeat.

The first physical case completed with 2,001 body observations, 400 neural
frames and 36 delayed decisions. Independent reconstruction from recorded pose
reproduces every sensory input, neural state, query choice and applied command.
Five deliberately altered records remain rejected even after their archive
hashes are updated: wrong cue, command before query, invented neural state,
wrong sampled pose and wrong phase progress. Run this integration audit after
the first case exists:

```sh
python scripts/verify_feedback_faults.py
```

After the entire physical queue finishes, the research environment runs:

```sh
python scripts/closed_loop_report.py
```

The reporter refuses incomplete cohorts, independently replays all controllers,
checks the repeated physical/neural arrays and builds a staged release under
`output/closed-loop-release`. Review its figures and video before publishing.
The first video contains 200 frames at 25 fps: two simulated seconds shown at
quarter speed. The encoder expands the requested 480×360 image to 480×368 for
codec block compatibility; physical measurements use MuJoCo state, not pixels.
Results are exploratory, use one model/physics seed and contain no language
weights or new learning during physical trials.
