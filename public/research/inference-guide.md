# Try the six WikiText comparison models

FLM by Kuber Mehta. This bundle runs the already-published FLM, GRU and decoder
transformer checkpoints on your own prompts, locally on a CPU. Each architecture
has two training seeds, 42 and 43. They share a 4,096-token vocabulary, the same
sampled training windows and 6,000 updates, with about 600,000 trainable parameters.
They were trained from scratch on the official WikiText-2 raw training partition.

These are small text continuation models. They have not been instruction-tuned.
Repetition, broken words and invented statements are retained in the published
examples. On the complete 60-article test partition, mean bits per UTF-8 byte
were **1.9744 for FLM, 1.9049 for GRU and 1.8767 for transformer**; lower is better.
FLM trails both baselines in this experiment. The new graph controls and BabyLM
models belong to continuing studies and are not included here.

## Download and install

Download [the comparison bundle](https://flm.kuber.studio/research/wikitext2-inference.zip)
and unzip it into a new directory. The public
[release record](https://flm.kuber.studio/research/inference-release.json) gives its
SHA-256 and byte size. The archive contains its own Python runtime source, so the
private source repository is not needed for inference.

From the extracted directory, create a Python environment. The release was
verified on Windows with **Python 3.10.11, PyTorch 2.8.0 CPU, NumPy 2.2.6,
SciPy 1.13.1 and tokenizers 0.22.2**. Linux CPU installation uses the same package
commands; that platform has not been independently tested for this release.

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` in a Linux shell, then install:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
```

The CPU wheel command follows the official
[PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/#v280).
This runtime needs neither torchvision nor torchaudio. A GPU, API key and raw
training corpus are unnecessary for generation.

## Compare the same prompt

Run each command from the extracted directory:

```sh
python -X utf8 -m flm.inference --model flm --prompt "The history of science"
python -X utf8 -m flm.inference --model gru --prompt "The history of science"
python -X utf8 -m flm.inference --model transformer --prompt "The history of science"
```

JSON output includes the original selected-checkpoint identity, sampling settings,
continuation, token IDs and decoded byte count. The defaults reproduce the
published sample protocol: training seed 42, sampling seed 17, temperature 0.8,
top-k 40 and at most 128 output tokens. The other fixed prompts are:

- `In the summer, the village`
- `The small animal moved through`
- `Language is a way to`

Set `--training-seed 43` to compare the second trained initialization. Keep the
prompt and settings equal across architectures. A sampling seed is distinct from
the training seed and does not select a different trained model.

```sh
python -X utf8 -m flm.inference --model flm --training-seed 43 --prompt "A small animal" --sampling-seed 29 --temperature 0.8 --top-k 40 --tokens 256
```

`--temperature 0` uses greedy argmax; in that mode top-k and the sampling seed
do not affect the output. Generation stops at the end-of-document token or the
requested token limit (1 to 1,024). The beginning-of-document token and byte pieces
containing control characters other than tab/newline are excluded from sampling.
The published likelihood scores use the model's full distribution without this
generation-only filter.

Each invocation resets state, prepends the beginning-of-document token and feeds
the entire prompt in chunks of at most 96 tokens. FLM and GRU carry their recurrent
state; the transformer carries its bounded attention cache and positional offset.
The CLI uses one CPU thread to match the published sampling engine. Exact sampled
text is checked in the tested environment; other PyTorch versions or platforms
may introduce numerical differences that alter stochastic continuations.

## What is in the archive?

- Six model-only checkpoints with learned tensors, fixed buffers and configuration.
- The measured 1,024-neuron graph subset and its attribution/provenance card.
- The exact train-only BPE tokenizer and dataset/tokenizer cards.
- The minimal runtime source, pinned inference dependencies and component licenses.
- The frozen original selection record, published test summary and training protocol.
- `bundle.json`, which inventories file sizes, SHA-256 hashes and model identities.

Optimizer state, training RNG, raw corpora and user conversations are omitted.
This bundle supports inference; it cannot resume the original training runs.
Original code and model weights use the repository's MIT license. The imported
brain graph retains CC BY 4.0 attribution; see `licenses/DATA-ATTRIBUTION.md`.
Corpus source/license details remain in the included dataset card.

Exported checkpoint files differ from the larger training checkpoint files.
The manifest preserves the **original selected-checkpoint SHA-256** and separately
records the **exported file SHA-256** and a **canonical tensor/buffer state hash**.
The exporter compares every restored tensor and fixed buffer exactly and replays
all four published prompts for each of the six models before writing the release.

Every load verifies the bundle files, graph, tokenizer and matching runtime source.
A modified or incomplete bundle is rejected before its checkpoint is loaded.
These hashes detect accidental changes; obtain the archive and release record
from the project to establish their provenance.

ChatFLM's browser runtime remains FLM-only. This Python bundle is the way to try
GRU and transformer with your own prompts. It is a continuation interface,
separate from the browser's conversation storage, adaptation and 3D views.

## Reproduce training and scoring

The included `docs/WIKITEXT-PROTOCOL.md` describes the matched experiment, selection
rule and split discipline. The published `public/research/test-results.json`
contains the completed comparison. Training acquisition, fitting and full-article
scoring commands are in the [repository README](https://github.com/Kuberwastaken/flm).
Repository access is required to run that full training pipeline; it is not part
of this compact inference archive. Trying new prompts does not change the frozen
test results or provide evidence of a general anatomical advantage.
