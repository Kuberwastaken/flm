# Articulated body

`public/body/model.json` and the 39 binary STL meshes derive from NeuroMechFly via the public Neural Canvas asset release pinned at `776d115ee5aa934578a87fd6d260d138084f59c1` in the Xenova/fruit-fly-simulation Hugging Face Space. The original body asset chain includes NeLy-EPFL/fly-svg-maker revision `152506d3471646f009480c81f34aefbaec29a6e5` and FlyGym assets. The body is a female micro-CT-derived mechanical model; the graph is from the male CNS dataset. They are different specimens.

`web/body/fk.js` and `web/body/stl.js` preserve the upstream forward-kinematics and STL parsing implementation. Applicable upstream MIT and Apache-2.0 notices are included alongside this file. Three.js carries its own MIT notice. These imported files are distinct from original FLM code.

The ChatFLM body view maps aggregate recurrent-state values to a few joint offsets for illustration. It does not implement a learned motor policy, physics, proprioception, flight, walking, or biological language learning. No connection to a living fly's behavior is claimed.

## Compiled physical-replay assets

`public/body/recorded-food/` is a separate export of all 69 visual meshes from the
pinned FlyGym 2.1.0 / MuJoCo 3.9.0 model used by the physical food experiment.
`experiments/embodiment/food_replay_assets.py` records the compiled geometry IDs,
body IDs, local mesh transforms, source hashes and environment. Its packed
vertices and triangles are verified exactly against the compiled arrays.
These NeuroMechFly-derived assets retain the body-source attribution and
component notices above. The different male neural graph and female body remain
different specimens. The export adds no learned behavior or physical trials.

The recorded-state viewer must use compiled geometry transforms rather than
the API's unresolved head-body lookup. See
[`PHYSICAL-STATE-SEMANTICS.md`](../research/physical-state-semantics.md) for the lookup
defect and the distinction between cached sensor poses and reconstructed `qpos`
geometry. The existing illustrative chat body remains separate.
