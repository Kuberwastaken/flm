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
