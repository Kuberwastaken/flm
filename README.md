# FLM · Fly Language Model

**[Open ChatFLM](https://flm.kuber.studio)** · [Methods](docs/ARCHITECTURE.md) · [Research basis](docs/RESEARCH.md) · [Comparison protocol](docs/WIKITEXT-PROTOCOL.md)

FLM is a small recurrent language model whose directed connections are selected from a fruit-fly connectome. ChatFLM runs the trained model locally in a browser, displays its actual recurrent state at anatomical neuron positions, and lets you inspect predictions, silence neurons and adapt a separate readout to your own text.

The main experiment uses **WikiText-2 raw**, a standard written-language corpus with 600 training articles and 2.05 million training words. A 4,096-token lossless byte-pair vocabulary is learned only from those articles. FLM, a GRU and a basic decoder transformer use the same tokenizer, sampled training windows and approximately 600,000 parameters. The earlier AMI meeting-transcript model remains available as a separate dialogue experiment.

This is a working research prototype, not an instruction-following assistant. The current matched validation curves do **not** establish an FLM advantage over the baselines. See the site's Research view for measured curves, matched-update controls and unedited fixed-prompt continuations. The second training seed and final test-set evaluation are still pending; validation scores must not be described as final test results.

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

Text and adaptation stay in the browser. Conversations are saved locally and can be exported/imported. Learning changes a separate output adapter and persists only when explicitly saved or exported. Adapters are tied to a checkpoint hash. A model switch reloads the workspace, so save session learning first. Browser storage can be unavailable or full; file export remains available. The optional 3D views need WebGL; text inference runs in a CPU worker.

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
