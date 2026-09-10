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
