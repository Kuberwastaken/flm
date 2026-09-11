# A scripted odor rule reaches the source but ignores reward reversal

Kuber Mehta · 11 September 2026. All seven declared physical trials completed,
with no simulation failures. **This is a scripted reference with zero training
updates and no language model.** The [prior declaration](food-approach-reference.md)
fixes the controller, environments, sensor access, physical budget and outcome
definitions. Its source-bound identity preceded every trial.

![Actual thorax paths and every declared first-contact outcome. The scripted rule follows odor A despite sugar reversal; missing odor and straight walking time out.](figures/food-approach.png)

## What happened in the simulator

The reference samples bilateral odor A and steers the existing calibrated gait
controller. It stops on any-source contact. It receives neither source positions
nor target-bearing information. Odor B and sugar contact are available in the
six-channel interface but deliberately unused by this fixed reference rule.

| Declared condition | First sampled contact | Source | Sugar intensity at that contact |
|---|---:|---|---:|
| Odor A on the left | 0.65 s | A | 1 |
| Odor A on the right | 0.68 s | A | 1 |
| Sugar identities reversed | 0.65 s | A | 0 |
| Both sources neutral | 0.65 s | A | 0 |
| Odor removed | No contact by 2 s | — | — |
| Straight reference | No contact by 2 s | — | — |
| Exact-seed repeat of A-left | 0.65 s | A | 1 |

No trial contacted source B. Reversed sugar therefore produces the wrong
approach for obtaining sugar: the policy still follows odor A. Neutral-source
contact is retained as such, not counted as sugar success. The two timeouts are
censored observations at the full two-second budget, not excluded failures or
two-second contact events. All outcomes and figure values are available in the
[complete summary](food-approach/summary.json) and
[CSV](figures/food-approach.csv).

The left/reversed/neutral/repeat conditions have exactly equal recorded body
positions, thorax orientations, generalized positions and descending commands.
The exact repeat also reproduces all sensor and contact arrays. Removing odor
produces exactly the same physical path as the straight reference. Each
simulation was initialized and run independently; the figure shares a line
only after verifying those array equalities. These are reproducibility and
input-isolation checks, not independent biological replications.

## What this establishes

Within these declared situations, the engineered odor sensors can drive an
actual source approach through the simulated body and upstream gait controller.
The approach changes when odor is absent or the source layout is mirrored.
The reward reversal exposes the fixed rule's limitation: reward identity has
no effect on its action history. It has no learned association to reverse.

The records do not demonstrate a neural learning advantage, an anatomical
advantage, language-to-behavior transfer, learned gait, ingestion or a biological
feeding response. Virtual foot-origin contact with a cylinder is the operational
endpoint. It is sampled at 100 Hz and does not establish the first continuous-time
physical collision. The displayed trajectories trace the thorax, which need not
enter a patch for a foot origin to contact it.

These seven hand-declared cases use one gait seed. They do not estimate
performance across randomly sampled layouts, independent animals or unseen
learned-task conditions. No confidence interval or aggregate success-rate claim
is supported by treating these diagnostic cases as independent trials.

## Recorded evidence and verification

The [identity](food-approach/identity.json) binds local numerical
sources, the prior declaration, package versions and installed FlyGym body API
source hashes. All seven trials run 20,000 physics steps at 0.0001 s and retain
201 observations each: **140,000 physical steps and 1,407 recorded observations**.
The run used one numerical thread in an isolated environment, at below-normal
Windows priority. The active language trainer and its numerical sources were
not modified.

The [independent audit](food-approach/audit.json) recomputes every
sensor vector from recorded body geometry with scalar math, then reconstructs
every scripted descending command, including the latched contact stop. It also
recomputes contact latencies, source identities, censoring, physical metrics and
all four paired-array comparisons. It imports neither FLM nor FlyGym and does
not independently rerun the physics. This is a boundary on the evidence: a
record replay cannot prove that supplied physical positions are true.

Five policy tests cover turn direction, bounded commands, unused reward/odor
inputs, stopping, counterfactual field invariance and malformed inputs. Four
audit tests use explicitly synthetic records to reject changed commands,
invented reward signals, altered sensor geometry, masks or clocks, invented
contact success and missing budgets; an empty failed trial remains a failure.
The synthetic fixtures are not extra physical trials.

Download the [complete trajectory and audit archive](food-approach-records.zip).
It includes all seven original NPZ arrays, their trial JSON records, the complete
identity and summary, the scalar sensor auditor, standalone outcome auditor,
prior declaration and bound numerical source files. After extracting it into
a fresh directory with NumPy installed:

```powershell
python audit_food_approach.py . --output fresh-audit.json
```

The [release record](food-approach/release.json) contains hashes for
the archive and every figure artifact. The archive is a reference-record
release; recreating physical motion additionally requires the pinned isolated
FlyGym environment. No trained food checkpoint exists.

## Next learning comparison

The [food-response plan](food-response-plan.md) now has a working scripted
sensor-to-action reference and an explicit reward-reversal failure. The paired
random-core versus language-trained-core comparison still needs a frozen
adaptation task, independent evaluation layouts, fitted interfaces and measured
retention. It must distinguish learning a new readout from modifying recurrence
and report whether it does better or worse than this simple reference.
The selection-language comparison retains priority after the fixed BabyLM queue.
