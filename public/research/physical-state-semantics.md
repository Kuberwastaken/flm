# Recorded sensors and reconstructed body geometry

Kuber Mehta · 11 September 2026. This post-hoc implementation note identifies
two distinctions needed for a faithful physical replay. It does not revise any
frozen protocol, original trajectory, sensor input, neural command or outcome.

## An unresolved head lookup

The pinned FlyGym 2.1.0 body API enumerates 69 segment names. In the compiled
model used here, its lookup for `flm_body/c_head` returns `-1`. The API passes
that index into the NumPy body-position and body-quaternion arrays. Consequently,
the reported `c_head` row reads the final compiled body, `flm_body/rh_tarsus5`.
It is **not a valid head position**. The other 68 enumerated body names resolve.

The [compiled lookup inventory](food-core-physical/body-assets.json)
records every requested name and resolved ID. All **nine** body lookups used
for the food observations and thorax outcomes resolve correctly: two antennae,
six feet and the thorax. The head row
does not enter these sensors, the core, the gait controller input or the
reported thorax paths and contact outcomes.

The [native inspection](food-core-physical/inspect_food_body_semantics.py)
checked all six completed initial-seed-42 cases, covering 1,206 observations.
Every recorded head row exactly equals the recorded right-hind tarsus-5 row.
The [inspection record](food-core-physical/body-semantics.json)
preserves file hashes, the source identity and all per-case measurements.
The 24-case comparison was still running at this inspection; this is a method
check on those six cases, not its complete result.

The head's **visual mesh does exist**, attached to another compiled body.
The [new body asset exporter](food-core-physical/food_replay_assets.py)
therefore resolves visual geometry names directly and exports all 69 compiled
meshes. Every exported vertex and triangle is decoded and compared with its
compiled source. A recorded-state renderer must use each mesh's actual geometry
transform, rather than turning the invalid head body row into a displayed pose.
The existing illustrative chat body is a separate asset and rendering path.

## Cached kinematics and integration state

The physical runner records `qpos` and the cached `xpos`/`xquat` returned by the
body API after `mj_step`. These belong to different stages: `qpos` has advanced,
while the cached derived quantities describe the preceding position stage.
MuJoCo documents this behavior explicitly in
[“Consistency in mjData”](https://mujoco.readthedocs.io/en/stable/computation/index.html#consistency-in-mjdata).
The installed simulation uses Euler integration with a 0.0001-second step.

Recomputing kinematics from a saved `qpos` therefore need not exactly reproduce
the cached body positions that supplied that sample's sensors. Across the
inspected 1,206 observations, the largest Euclidean difference among the 68 valid
body rows was **0.011728 mm** (rounded upward). The record also retains per-case
coordinate and thorax-quaternion differences. This measured value is specific
to the inspected cases; it is not a universal tolerance for other runs.

The distinction does not make the recorded sensor-to-command replay ambiguous:
that audit uses the original cached positions and original sensory inputs.
Reported contact times remain **sample-clock timestamps**, evaluated every
0.01 seconds. No contact was recomputed or relabeled from the later `qpos` stage.
Recomputing geometry for display is a separate operation and must be labeled
as reconstruction from recorded generalized coordinates. A future experiment
can choose a different sensing stage only through a new declared protocol.

## What this does and does not establish

The inspection ran kinematics on saved coordinates, with zero new physics steps
and zero training updates. It found a concrete body-lookup defect and measured
the integration-stage difference. It does not validate feeding, anatomical
learning, language-to-food transfer or every aspect of the simulator. Original
records are retained, including the invalid head row, so the limitation remains
visible to anyone auditing or reusing them.
