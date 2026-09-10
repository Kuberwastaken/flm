# FLM · Fly Language Model

**[Open ChatFLM](https://flm.kuber.studio)** · [Methods](docs/ARCHITECTURE.md) · [Research basis](docs/RESEARCH.md) · [Comparison protocol](docs/WIKITEXT-PROTOCOL.md)

FLM is a small recurrent language model by Kuber Mehta whose directed connections are selected from a fruit-fly connectome. ChatFLM runs the trained model locally in a browser, displays its actual recurrent state at anatomical neuron positions, and lets you inspect predictions, silence neurons and adapt a separate readout to your own text.

The main experiment uses **WikiText-2 raw**, a standard written-language corpus with 600 training articles and 2.05 million training words. A 4,096-token lossless byte-pair vocabulary is learned only from those articles. FLM, a GRU and a basic decoder transformer use the same tokenizer, sampled training windows and approximately 600,000 parameters. The earlier AMI meeting-transcript model remains available as a separate dialogue experiment.

The continuing study adds the official **BabyLM 2026 10M- and 100M-word corpora**, verified locally across six spoken/written components. Its [registered protocol](docs/BABYLM-PROTOCOL.md) separates data diversity, model capacity and training exposure. Measured neural interventions and a real, separately calibrated NeuroMechFly physics environment support later learning and behavior studies. See the [continuing research program](docs/RESEARCH-PROGRAM.md).

**Current priority: does the measured wiring help language?** The
[language control study](docs/LANGUAGE-TOPOLOGY-PROTOCOL.md) trains three
independently rewired graphs with both original seeds and two models retrained
without slow state. It preserves the original WikiText data, initialization,
sampling and budget. BabyLM training is paused at saved checkpoints while these
controls run. A degree-matched anatomical advantage remains unestablished.

The [completed 60-run cue/context study](docs/WIRING-RESULTS.md) has mixed
outcomes, including conditions where rewiring performs better. All seeds,
checkpoints, prediction panels and gradient diagnostics are retained. It is a
separate artificial-task experiment, not evidence of a language advantage.

This is a working research prototype, not an instruction-following assistant. The completed two-seed WikiText test comparison gives mean losses of **1.9744 bits/byte for FLM, 1.9049 for GRU and 1.8767 for transformer**; lower is better. FLM trails both neural baselines under this protocol. The Research view includes full article scores, paired intervals, measured inference costs and unedited fixed-prompt continuations. BabyLM training is a separate, continuing experiment.

## What is different?

FLM propagates a token through a fixed, signed recurrent graph. Learned edge magnitudes, bounded update rates and fast/slow state determine its response. A conventional transformer instead computes content-dependent attention over token representations. FLM's 1,024-neuron core carries 8 KiB of float32 state independent of sequence length; this is lossy memory and does not imply superior prediction, energy use or biological realism.

The input embeddings, output projection and optimizer are engineered language-learning components. The anatomy is not a recovered pretrained intelligence. This implementation is a differentiable rate network, not a spiking simulation. The body view uses authentic articulated NeuroMechFly geometry, but its pose is an illustrative mapping from aggregate state, not a learned motor policy. The brain and body come from different-sex specimens.

## Run the interface

Requires Node.js 22.12 or newer. The repository includes browser checkpoints, tokenizer and attributed anatomical/body assets; no hosted model API or key is needed.

```sh
npm ci --ignore-scripts
npm run dev -- --port 5180
```

Open the URL printed by Vite. Use `npm run build` and `npm run preview -- --port 5181` to inspect a production build. GitHub Actions publishes `dist/` to Pages. `public/CNAME` points to `flm.kuber.studio`; the repository remains private while the generated website is public.

Text and adaptation stay in the browser. Conversations are saved locally and can be exported/imported. Learning changes a separate output adapter and persists only when explicitly saved or exported. Adapters are tied to a checkpoint hash. A model switch reloads the workspace, so save session learning first. Exports expose their complete JSON for copying when native file downloads are unavailable. The optional 3D views need WebGL; text inference runs in a CPU worker.

## Try FLM, GRU and transformer locally

The [six-model inference bundle](https://flm.kuber.studio/research/wikitext2-inference.zip)
contains the published WikiText checkpoints for both training seeds, their shared
tokenizer, measured graph subset and a standalone Python runtime. It runs without
access to the private repository or training corpus. Follow the
[installation and generation guide](docs/INFERENCE-BUNDLE.md), then compare:

```sh
python -X utf8 -m flm.inference --model flm --prompt "The history of science"
python -X utf8 -m flm.inference --model gru --prompt "The history of science"
python -X utf8 -m flm.inference --model transformer --prompt "The history of science"
```

Run these commands inside the extracted bundle after installing its dependencies.
The [release record](https://flm.kuber.studio/research/inference-release.json)
identifies all six original selected checkpoints and the archive's SHA-256.
Every exported tensor and fixed buffer matches exactly; all 24 published
continuations replay under the tested CPU environment. This is local continuation
inference. ChatFLM's browser engine remains FLM-only.

## Reproduce the language experiment

Requires Python 3.10 or newer and a PyTorch-supported environment. The current run uses PyTorch 2.8.0 on CPU. Raw corpora and training checkpoints are ignored by Git.

```sh
python -m pip install -e ".[language]"
python -m flm.wikitext
python -m flm.tokenizer
python -m flm.language_suite
```

Acquisition verifies the pinned source revision, sizes and SHA-256 hashes. Article grouping preserves every source character, keeps the official partitions and audits document identities. Tokenizer training never reads validation or test text. The subsequent encoding stage transforms each split with the frozen tokenizer. Review [the dataset card](data/cards/wikitext2.json), [tokenizer card](data/tokenizers/wikitext2-4096/tokenizer-card.json) and [registered protocol](docs/WIKITEXT-PROTOCOL.md).

The suite runs seeds 42 and 43 for FLM, GRU and transformer, 6,000 updates each. It is CPU-intensive and may take hours on a laptop. Run only one writer for a given output directory. Completed artifacts are checked before skipping; incomplete runs resume from their saved optimizer and RNG state. To run one model explicitly:

```sh
python -m flm.language_train --variant flm --seed 42 --output runs/wikitext2/flm-s42
python -m flm.language_train --variant flm --seed 42 --output runs/wikitext2/flm-s42 --resume runs/wikitext2/flm-s42/last.pt
```

Use the second command only to resume an existing, stopped run. Keep the original schedule, seed and thread count. Changing the protocol requires a new output directory.

```sh
python -m flm.language_report --sample-step 1000
python -m flm.ngram --data data/processed/wikitext2 --output runs/wikitext2/ngram
python -m flm.export_lexical runs/wikitext2/flm-s42/best.pt
```

The report consolidates validation artifacts and generates fixed-prompt comparisons from numbered checkpoints. The n-gram reference tunes smoothing on validation and is not parameter-matched. The exporter creates a small tied-embedding browser package and independent PyTorch parity fixtures. Its optional `--anatomy-source` accepts the verified full soma array to include gray anatomical context; active neuron coordinates always come from the selected graph.

The tracked compact graph is sufficient for training. To reproduce it from the complete upstream graph:

```sh
python -m flm.acquire_graph
python -m flm.graph --source data/processed/connectome --output work/reproduced-graph
```

The acquisition verifies 166,700 neurons, 25,582,938 directed neuron-pair edges and 124,177,617 contacts. The trained subset has 1,024 neurons and 76,130 retained edges. [Graph provenance](data/graphs/central-1024/graph-card.json) records selection and source identities. Training the full connectome is a separate scaling stage, not demonstrated by this compact release.

## Larger corpora and physical behavior

The BabyLM pipeline preserves original UTF-8 text, learns its vocabulary only from
the 10M training set, audits normalized line overlap and stores source-indexed
blocks in verified memory-mapped files. Raw corpora stay local. The publisher's
training-repository MIT metadata does not replace component rights.

```sh
python -m flm.babylm acquire
python -m flm.babylm_audit
python -m flm.babylm_prepare
python -m flm.babylm_pipeline
```

The suite requires the completed WikiText runtime report first, then runs the
12 registered combinations serially: two corpora, two seeds and three models,
12,000 updates each. It can take many hours on a CPU. `--scales 10m --seeds 42
--variants flm` runs a specified portion of the same protocol. All comparisons
share the 10M-fitted tokenizer and a frozen 48-block validation panel. Official
test scores and an overlap-filtered sensitivity analysis are separate from
training progress. Do not treat the 100M dataset label as a 100M-parameter model.

`babylm_pipeline` runs or resumes training, freezes all twelve validation-selected
checkpoints, evaluates every official test block, computes the predeclared
overlap-filtered and per-source scores, and generates all 288 fixed-prompt
continuations. Start it only after any existing training process has stopped;
there must be one training writer. To run training alone or a specified portion,
use `python -m flm.babylm_suite` with the selection flags above. Completed training
is reused; a failure stops dependent stages. Full test evaluation can also take
hours on a CPU, with atomic completed-batch caches for recovery.

The individual follow-on commands are `python -m flm.babylm_test` and
`python -m flm.babylm_samples`. They refuse incomplete studies or changed
checkpoint/data identities. `python -m flm.babylm_report` publishes a dated
validation snapshot without reading model test losses. The website compares
only updates shared by all three architectures, and can isolate each source
component. See the [evaluation declaration](docs/BABYLM-EVALUATION.md) and
[original prompt panel](data/prompts/babylm-original.json). Repetition measurements
are descriptive; they do not score truth or replace human assessment.

The physics environment uses Python 3.12 with `requirements-embodied.txt`; the
exact measured environment is recorded in `requirements-embodied-lock.txt`.
Run `experiments/embodiment/calibrate.py` inside that isolated environment to
reproduce the designed walking controls. See its [README](experiments/embodiment/README.md).
The research page includes actual simulated videos and trajectory data. The
initial calibration has no FLM connection. A subsequent 40-case assay connects
recorded choices from a trained sensory network to the calibrated controller;
the browser body view remains a separate illustrative state-to-pose mapping.

## Forward learning and learned physical choices

A separate 256-neuron network compares BPTT, fixed-core readout learning,
supervised forward eligibility, a no-history control and reward-modulated
eligibility. All five receive the same initialization and sensory stream within
each of three seeds. The task learns, reverses and restores a delayed cue rule
over 900 updates. This is a controlled rate-network experiment, with engineered
inputs and outputs; it is not language-to-motor transfer or a biophysical model.

```sh
python -m flm.behavior_study
python -m flm.behavior_report
python scripts/behavior_figures.py
```

The committed [protocol](docs/LOCAL-LEARNING-PROTOCOL.md) precedes fitting. All
15 completed runs retain diagnostic predictions, neural states, checkpoints and
stimulus hashes. On the final 48-frame delay probe, mean accuracy is 100% for
BPTT, 86.46% for the fixed core, 83.33% for supervised eligibility and 50% for
both no-history and reward eligibility. Three seeds and one simple task do not
establish a general ranking. The strong fixed-core result means this task alone
does not demonstrate a benefit from learning recurrent wiring.

Using the isolated physical environment, run
`python experiments/embodiment/learned_choice.py --video`. Then run
`python scripts/physical_choice_report.py` in the research environment. The
assay independently simulates both cues at four predetermined checkpoints for
all five methods, using seed 17. All 40 physical headings follow their chosen
command; 33 neural choices match the task rule, including untrained checkpoints.
These are two repeatable motor commands, not 40 independent skills. The network
chooses before replay; the designed gait controller handles leg motion and
contact feedback. Online proprioception, learned balance and transfer from a
language-trained core remain future experiments.

The [four-page working note](public/research/local-learning.pdf) derives the
normalized-edge eligibility approximation and reports failures, storage costs
and reproduction details. See [paper build instructions](papers/README.md).

## Online pose feedback

The [closed-loop assay](docs/CLOSED-LOOP-PROTOCOL.md) is complete: 27 physical
conditions and one exact repeat, with independently replayed sensory inputs,
neural states and delayed commands. It crosses untrained/BPTT/fixed-core/
eligibility checkpoints with live or frozen pose in three waypoint scenarios,
plus a scripted reference. The three trained controllers have identical physical
paths to that reference in all scenarios. Live feedback lowers their mean
angular error from 96.41/93.91/100.50 degrees to 7.68/7.26/16.27 degrees for
positive/negative/switch targets. This establishes a working cue-to-body
interface, without evidence that training recurrent dynamics was necessary.

All 27 conditions produce seven distinct recorded body paths. The single model
and physics seed do not support population intervals. These are fixed sensory
models using simulator ground truth, an engineered deadband and a designed gait;
they do not use language weights or learn motor behavior during the assay.
The [four-page note](public/research/closed-loop.pdf) reports every condition.
The [45.9 MB standalone records archive](https://flm.kuber.studio/research/closed-loop-records.zip)
contains all trajectories, controllers, parity fixtures, source and licenses.
See [audit and fresh-physics instructions](docs/CLOSED-LOOP-REPRODUCTION.md).

The separate language topology comparison remains the priority: two of eight
new controls have completed their 6,000-update budgets. Their test scores remain
locked until all runs and checkpoint selections are complete. BabyLM stays paused.

## Verification and layout

```sh
python -m unittest discover -s tests -p "test_*.py"
npm test
npm run build
```

Tests cover causal streaming, document resets, graph constraints, exact resumed updates, byte accounting, tokenizer parity, independent browser inference, adaptation, worker cancellation and body geometry. Browser observations and their limitations are recorded in [the validation log](docs/BROWSER-VALIDATION.md).

| Location | Contents |
|---|---|
| `flm/` | Acquisition, graph selection, models, training, export and reports |
| `web/` | Inference worker, tokenizer, learning, UI and 3D views |
| `data/cards/`, `data/tokenizers/`, `data/graphs/` | Versioned provenance and compact assets |
| `public/models/`, `public/research/` | Browser checkpoints, measured curves and generated samples |
| `docs/`, `tests/` | Research decisions, protocols and verification |
| `runs/`, `data/raw/`, `data/processed/` | Local, untracked training and corpus intermediates |

Original implementation: MIT. Imported components retain their own licenses. Brain data and AMI transcripts use CC BY 4.0; WikiText publisher metadata lists CC BY-SA 3.0 and GFDL while its prose links another license version. That discrepancy is preserved in the card. Raw corpus text is not redistributed. See [component notices](licenses/) and the [published attribution page](https://flm.kuber.studio/licenses/).

Contributions should preserve provenance, add meaningful checks for changed behavior and report unsuccessful experiments. Commit coherent changes sequentially. Never present scripted output or decorative neural activity as inference, compare scores across incompatible tokenizers, or claim biological/transformer superiority from a few samples.
