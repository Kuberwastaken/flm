# Reward-modulated action readout: implementation before food adaptation

Kuber Mehta · 11 September 2026. This is tested learning machinery, **not a
frozen physical training protocol or a food-adaptation result**. It changes no
completed experiment. Complete selection groups retain training priority after
the fixed BabyLM queue.

## The question this isolates

The [unadapted physical reference](food-core-physical-results.md) includes
wrong-source contacts and a sugar contact with odor removed. Reaching a patch
does not show that the language core has learned an odor/reward association.
A downstream learner must receive matched task exposure before comparing its
change from baseline across initial and language-trained cores.

The first implemented learner freezes the sensory projection, recurrent edges,
time constants and normalization. It learns only the three-action linear
readout from the core's 256 normalized pooled fast/slow features: **771 weight
and bias entries**. This is narrower than jointly adapting the sensory and
action interfaces. That distinction must be a registered factor, rather than
calling either one general task adaptation. A later sensory-adapter or
recurrent-learning condition needs its own matched controls.

The code lives in [food_readout_learning.py](food-readout/food_readout_learning.py).
It accepts feature vectors; it cannot change the recurrent core, inspect source
coordinates or read reward identity before the caller supplies terminal reward.
It copies its initial weight/bias arrays. Its episode state and complete trace
can be serialized and resumed exactly. The [episode runner](food-episode-runner.md)
now binds it to the frozen core, action stream and physical environment interface.
The [paired schedule executor](food-adaptation-schedule.md) now checks complete
training inventories and fixed-checkpoint evaluation. Official study
registration and cost/prerequisite checks remain pending. Interrupted episodes
restart original conditions; live simulator/gait checkpoint resumption remains
unimplemented.

## Learning rule

For fixed core features $z_t$, the readout is

$$p_t=\operatorname{softmax}(Wz_t+b),\qquad a_t\sim p_t.$$

The score vector $s_t=\operatorname{onehot}(a_t)-p_t$ accumulates eligibility:

$$E^W_t=\lambda E^W_{t-1}+s_tz_t^\top,\qquad
E^b_t=\lambda E^b_{t-1}+s_t.$$

Parameters remain unchanged during the episode. After terminal reward $R$:

$$W\leftarrow W+\eta(R-\bar R)E^W_T,\qquad
b\leftarrow b+\eta(R-\bar R)E^b_T.$$

Only afterward is the baseline updated:
$\bar R\leftarrow\bar R+\rho(R-\bar R)$.
The baseline starts at zero and uses earlier completed episodes. Failed
episodes are explicitly aborted without changing weights or the baseline;
their failure records must remain in the coordinator's inventory.

With $\lambda=1$, the trace sums the trajectory log-likelihood gradients.
Multiplying by terminal reward minus an action-independent baseline gives the
episodic likelihood-ratio policy-gradient estimator. This follows established
[REINFORCE work (Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696).
The expectation claim assumes on-policy action sampling and environment
transitions without direct readout-parameter dependence. A single update can
have high variance and need not improve reward.

The implementation also supports explicitly chosen $0\leq\lambda<1$ and
$\lambda=0$ controls. These decay or remove earlier action credit; they are
**not the same unbiased terminal-return estimator**. There is no undisclosed
normalization by episode length, entropy bonus, gradient clipping, weight decay
or critic. Such additions would change the method and need to be declared.

Uniform draws are supplied by the caller, allowing matched random-number
streams across policies without using NumPy's global RNG. Readout weights,
probabilities and traces use float64. Existing frozen core inference remains
float32; this new policy arithmetic is not bitwise identical to the earlier
float32 head. The pending training protocol must fix the action clock, sampling,
motor mapping and new initial-condition evaluation. The previous continuous
probability-to-drive trajectories cannot be reused as sampled-policy baselines.

## Biological connection and limits

The trace combines a feature term and action-dependent score, with later reward
modulating its effect. This has the general structure that motivates studies
of three-factor plasticity. [Frémaux and Gerstner (2016)](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full)
review how activity-dependent eligibility and modulatory signals can interact,
while emphasizing distinctions between theoretical and physiological rules.
Our softmax action score is engineered. It is not measured dopamine, a spiking
synapse, a Kenyon-cell learning rule or a restoration of missing mushroom-body
pathways. The retained graph still has its documented truncation.

The same readout algorithm could act on fixed GRU, transformer or other features.
It is not exclusive to connectomes. Any FLM contribution must come from measured
comparisons of those features and their retained state, not from renaming this
established learning algorithm. No backpropagation through the core is required
when only the readout changes; that alone is not proof of biological fidelity.

## Checks completed and work still required

The [nine focused tests](food-readout/test_food_readout_learning.py) check trajectory
log-likelihood gradients against central finite differences, and enumerate all
nine paths of a small two-step environment whose second observation depends on
the first action. The expected full-history update agrees with the numerical
expected-reward gradient at three baseline values. Other checks cover history
removal/decay, lagged baseline updates, exact JSON resume, invalid inputs,
decision budgets, sampling with underflowed probabilities and transactional
failure on update overflow. These fixtures are not physical food trials.

```powershell
python -m unittest discover -s tests -p test_food_readout_learning.py -v
```

The [food-response plan](food-response-plan.md) still requires training-only cost
measurements, complete condition/seed inventories, fixed layouts and reward
counterbalancing, a new sampled-action physical reference, unseen evaluation
layouts, reversal/retention probes and a frozen analysis before official fitting.
Stopping on first contact must exclude later stopped actions from reward credit.
Correct/wrong contact, no contact and simulator failure must stay distinct.
Replay of new sampled actions on old recorded observations is only a numerical
check: it does not generate their counterfactual physical trajectory or provide
an on-policy learning experiment.
