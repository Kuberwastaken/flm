# Connecting action learning to physical episodes

Kuber Mehta · 11 September 2026. The sampled-action controller now connects the
frozen food-sensory core, reward-modulated readout and existing FlyGym gait.
**Food adaptation remains untrained.** The checks below establish executable
integration and repeatability, not a navigation result or a language benefit.
The fixed BabyLM comparison and subsequent neuron-selection study retain priority.

## Action and reward timing

[FoodEpisode](../flm/food_episode.py) owns an independent frozen core and a copy
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

With learning enabled, only the [episodic readout rule](FOOD-READOUT-LEARNING.md)
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
trajectory by replaying old sensory observations. The future study coordinator
must either resume complete physical/gait checkpoints or restart interrupted
episodes from their preserved original conditions and record those attempts.

## Physical integration checked

The [FlyGym adapter](../experiments/embodiment/food_learning_environment.py)
uses the unchanged calibrated body factory, six-channel field and hybrid gait.
It keeps diagnostic body state separate from policy inputs and applies only
the declared sampled drive intervals. Records retain actual generalized
positions, cached body positions, thorax rotation, sensor geometry, neural
states and action draws. The existing [body-state limitations](PHYSICAL-STATE-SEMANTICS.md)
still apply; this adapter does not repair or relabel the historical head row.

The [integration command](../experiments/embodiment/verify_food_episode.py)
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

The [verification record](../reports/food-episode/preparation.json) and
[complete smoke records](../public/research/food-episode-smoke.zip) preserve the
checks and trajectories. The archive's `-1` terminal action and uniform values
mean no action/draw; they are not invalid samples produced by the policy.

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
# Pinned embodiment environment; fresh output directory, learning disabled:
.venv-embodied/Scripts/python.exe experiments/embodiment/verify_food_episode.py `
  --bundle work/food-core-interface-v1 --output work/new-food-episode-smoke
```

Use the [released core interface package](https://flm.kuber.studio/research/food-core-interface.zip)
and its preparation record. Source, tests and the record packager are included
in [the research source archive](../public/research/paper-source.zip).

The [food-response study plan](FOOD-RESPONSE-PLAN.md) still requires complete
conditions and seeds, train-only costs, a fixed adaptation budget, a full-horizon
sampled-policy baseline, unseen evaluation layouts, counterbalanced rewards,
reversal/retention probes, and an immutable training/evaluation coordinator.
Only after that experiment can the page show an evidence-based before/after
food-learning comparison. Existing language studies and physical references are
unchanged.
