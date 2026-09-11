# Connecting action learning to physical episodes

Kuber Mehta · 11 September 2026. The sampled-action controller now connects the
frozen food-sensory core, reward-modulated readout and existing FlyGym gait.
**Food adaptation remains untrained.** The checks below establish executable
integration and repeatability, not a navigation result or a language benefit.
The fixed BabyLM comparison and subsequent neuron-selection study retain priority.

## Action and reward timing

[FoodEpisode](food-episode/food_episode.py) owns an independent frozen core and a copy
of the incoming action readout. The caller supplies an explicit physics timestep,
steps per action, episode horizon, reward mapping, action seed, source/environment
binding and learning flag. No official study budget is chosen by this API.

At each observation it advances the existing float32 recurrent runtime using the
six sensory channels and computes normalized pooled features in float32. Readout
and probability arithmetic use float64. Right, straight and left are sampled with a private PCG64 stream;
their descending drives are respectively `[1.2, 0.4]`, `[0.8, 0.8]` and
`[0.4, 1.2]`, following the existing calibrated action order. The sampled drive
is held for the entire declared action interval through the designed gait.
Learning walking mechanics is outside this experiment.

At contact or the horizon, the observation updates the displayed recurrent
state but **does not draw or credit another action**. The preceding action
interval has already occurred. The terminal sample's zero drive is a stop
record, not an additional applied command. Thus a horizon of 200 physics steps
with 100 steps per action has three observations and two applied decisions.
Contact takes precedence when it occurs exactly at the horizon.
Contact is checked at observation boundaries, not every physics step; reported
arrival times therefore have the declared observation resolution.

The explicitly supplied reward mapping is linear in observed contact sugar,
with a separately supplied timeout reward. Source coordinates, names, task
phase and rewarded-side labels do not enter action selection. Contact intensity
retains environment precision for reward; the core makes its original float32
conversion for neural inference. A trial starting in contact is an invalid
starting situation and receives no update. A simulator failure is retained as a
failure, without being recoded as a timeout or a zero-reward training example.

With learning enabled, only the [episodic readout rule](food-readout-learning.md)
updates after a completed terminal outcome. Parameters are fixed within the
episode. With learning disabled, weights, bias, baseline and training counters
are restored unchanged; evaluation cannot train the policy or alter its future
reward baseline. Starting another episode can carry the learned readout while
resetting core state and the declared action stream.

## What resumption covers

Controller snapshots bind core weights, code, NumPy version, clock, reward rule,
source/environment identity, initial readout, sensory observations and sampled
actions. Restoration replays the complete controller history and checks the
resulting neural state, eligibility, readout, baseline and random-number state.
This catches inconsistent actions, altered credit traces and mismatched sources.
Failed attempts also replay as failures without a reward update.

This is **controller resumption only**. A resumed physical episode additionally
needs the exact MuJoCo and gait states at the same boundary. The current runner
starts a fresh environment per episode; it cannot reconstruct a physical
trajectory by replaying old sensory observations. The
[durable episode store](food-episode/food_episode_store.py) implements restarting
interrupted episodes from their preserved original conditions. The future
study coordinator must use that boundary consistently across its declared
inventory; live simulator/gait checkpoint resumption remains unimplemented.

### Durable results and interrupted attempts

Each stored episode binds its original core, readout, learning rule, action
seed, environment identity, clock, reward mapping and implementation. An
operating-system lock prevents concurrent writers; a leftover lock file alone
does not indicate a live process. A fresh environment must match the declared
identity before reset. The caller's original controller remains unchanged.

The store writes a start record before running physics. It preserves each
terminal trajectory, controller state, outcome and resulting readout in a
compressed record, published by atomic rename after flushing the staged file.
It reads and replays the committed record before returning the readout. A retry
of that episode returns the same result without constructing another simulator
or applying another update. This is a process-interruption boundary, not a
guarantee against every filesystem or power-loss failure.

An unfinished attempt has no accepted readout update. After acquiring the
released lock, the next call marks that attempt unfinished and starts a new
attempt with the original readout, neural state, action seed and gait seed.
Staged files and earlier attempts remain available. A hard interruption can
leave unrecorded physical steps; the journal explicitly marks that work and
reward unknown. It does not manufacture a partial trajectory. A durable
simulation failure is returned as a failure on later calls, rather than rerun
until successful. Invalid starts and cleanup failures also preserve the original
readout; any computed but discarded update remains visible in the record.

The [physical restart checks](food-episode/verify_food_episode_store.py)
covered all four released cores with learning disabled. For each core, a direct
20-ms run was compared with a fresh 20-ms restart after an intentionally
interrupted 10-ms attempt. **All four complete physical/neural/action records
matched exactly**. This comprises eight completed episodes, four preserved
interruptions and 2,000 post-warmup physics steps. All completed episodes timed
out at this short horizon. Subsequent cached calls ran no new physics.

The [46 original records](food-episode-restart-records.zip)
include every episode identity, attempt start, interruption marker and terminal
result. The [numerical verification](food-episode/restart-verification.json)
replayed all eight controllers and recomputed sensor geometry for all 24 saved
observations without new physics. Its [auditor and packager](food-episode/package_food_episode_store.py)
uses the same declared routines, not an
independent simulator. Fourteen additional [durability tests](food-episode/test_food_episode_store.py)
pass in both Python environments, including an actual child-process exit,
exclusive writers, interrupted writes, failed outcomes, damaged records and
single application of terminal updates. See the [test record](food-episode/restart-tests.json).

The [paired schedule executor](food-adaptation-schedule.md) now connects these
boundaries to complete training inventories and evaluation at fixed readout
checkpoints. Its checks use toy transitions; no official physical adaptation
schedule or budget is registered.

These checks establish short-run restart behavior. They do not establish
long-horizon reproducibility, navigation, food adaptation, or an advantage from
language training. A full study still needs its official condition membership,
cost/prerequisite checks, physical training/evaluation schedules and declared
statistical comparisons.

## Physical integration checked

The [FlyGym adapter](food-episode/food_learning_environment.py)
uses the unchanged calibrated body factory, six-channel field and hybrid gait.
It keeps diagnostic body state separate from policy inputs and applies only
the declared sampled drive intervals. Records retain actual generalized
positions, cached body positions, thorax rotation, sensor geometry, neural
states and action draws. The existing [body-state limitations](physical-state-semantics.md)
still apply; this adapter does not repair or relabel the historical head row.

The [integration command](food-episode/verify_food_episode.py)
completed eight short physical runs: each of the four released initial/language
cores, repeated with the same action and gait seeds. Each run recorded **20 ms
after the factory's warmup**, with three observations and two commands. Across
all eight runs there were **1,600 post-warmup physics steps**, no failures and
zero food-task updates. Every paired array matched exactly, including actual
physical and neural states; all core and readout weights remained unchanged.
Controller replay also matched exactly in the recorded NumPy environment.

All runs timed out without contact at this deliberately short horizon. Their
duration cannot establish search, arrival, biological competence or a useful
language-pretraining effect. They are not training-cost measurements or a new
sampled-policy navigation baseline.

The [verification record](food-episode/preparation.json) and
[complete smoke records](food-episode-smoke.zip) preserve the
checks and trajectories. The archive's `-1` terminal action and uniform values
mean no action/draw; they are not invalid samples produced by the policy.
The [record auditor](food-episode/audit_food_episode_smoke.py) reproduced sensor
geometry, controller states and decisions for all 24 saved observations, and
checked the four exact paired repeats. Its [audit record](food-episode/archive-audit.json)
adds no physics or learning. It uses the same declared numerical routines,
rather than an independently implemented simulator or controller.

## Reproduction and remaining work

Twenty-four unit checks pass in both the primary Python environment and the
isolated embodiment environment: fifteen controller/rollout tests and the nine
existing learning-rule tests. They cover credit timing, precise reward input,
invalid starts, failures, evaluation isolation, exact controller replay,
action-dependent toy transitions and untouched source/global RNG state. Toy
environments and gradient checks are not physical learning results.

```powershell
python -m unittest discover -s tests -p test_food_episode.py -v
python -m unittest discover -s tests -p test_food_readout_learning.py -v
python -m unittest discover -s tests -p test_food_episode_store.py -v
# Pinned embodiment environment; fresh output directory, learning disabled:
.venv-embodied/Scripts/python.exe experiments/embodiment/verify_food_episode.py `
  --bundle work/food-core-interface-v1 --output work/new-food-episode-smoke
# Numerical audit only, using the recorded NumPy version and source files:
.venv-embodied/Scripts/python.exe scripts/audit_food_episode_smoke.py `
  --archive public/research/food-episode-smoke.zip --bundle work/food-core-interface-v1 `
  --output work/new-food-episode-audit.json
# Fresh physical restart checks, all learning disabled:
.venv-embodied/Scripts/python.exe experiments/embodiment/verify_food_episode_store.py `
  --bundle work/food-core-interface-v1 --output work/new-food-restart-checks
# Or audit extracted records/ from the restart archive, without new physics:
.venv-embodied/Scripts/python.exe scripts/package_food_episode_store.py `
  --input work/extracted-food-restart/records --bundle work/food-core-interface-v1 `
  --output work/verified-food-restart.zip --report work/verified-food-restart.json
```

Use the [released core interface package](https://flm.kuber.studio/research/food-core-interface.zip)
and its preparation record. Source, tests and the record packager are included
in [the research source archive](paper-source.zip).

The [food-response study plan](food-response-plan.md) still requires complete
conditions and seeds, train-only costs, a fixed adaptation budget, a full-horizon
sampled-policy baseline, unseen evaluation layouts, counterbalanced rewards,
reversal/retention probes, and an immutable training/evaluation coordinator.
Only after that experiment can the page show an evidence-based before/after
food-learning comparison. Existing language studies and physical references are
unchanged.
