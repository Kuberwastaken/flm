# Articulated body

`public/body/model.json` and the 39 binary STL meshes derive from NeuroMechFly via the public Neural Canvas asset release pinned at `776d115ee5aa934578a87fd6d260d138084f59c1` in the Xenova/fruit-fly-simulation Hugging Face Space. The original body asset chain includes NeLy-EPFL/fly-svg-maker revision `152506d3471646f009480c81f34aefbaec29a6e5` and FlyGym assets. The body is a female micro-CT-derived mechanical model; the graph is from the male CNS dataset. They are different specimens.

`web/body/fk.js` and `web/body/stl.js` preserve the upstream forward-kinematics and STL parsing implementation. Applicable upstream MIT and Apache-2.0 notices are included alongside this file. Three.js carries its own MIT notice. These imported files are distinct from original FLM code.

The ChatFLM body view maps aggregate recurrent-state values to a few joint offsets for illustration. It does not implement a learned motor policy, physics, proprioception, flight, walking, or biological language learning. No connection to a living fly's behavior is claimed.
