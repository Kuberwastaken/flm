# Train and profile FLM on a Colab T4

**[Open the FLM notebook in Colab](https://colab.research.google.com/github/Kuberwastaken/flm/blob/main/notebooks/FLM_T4_Capacity.ipynb)** · [Inspect the notebook](https://github.com/Kuberwastaken/flm/blob/main/notebooks/FLM_T4_Capacity.ipynb) · [SmolLM/Gemma source audit](https://github.com/Kuberwastaken/flm/blob/main/docs/COLAB-NOTEBOOK-AUDIT.md)

The notebook is implemented and checked locally. **GPU execution is still pending.** It can attempt a 150M or 300M FLM using every acquired neuron, train from random weights on educational text, continue into human conversation training, and run a separate random transformer control. Its default trainer budget is 40 minutes, with setup and downloads additional. This is a capacity and learning probe, not a released larger ChatFLM or a claim of useful chat in an hour.

## What changed

The notebook adopts the practical workflow of small-model Colab recipes: limited runs, accumulated batches, response-only labels, memory tracking and saved checkpoints. The FLM computation stays recurrent. No pretrained SmolLM/Gemma weights, pretrained embeddings, teacher answers or attention layer enter FLM's language path. The optional transformer is separately initialized from a configuration and clearly labeled.

![Computed parameter allocation for the two full-graph configurations. GPU fit remains unmeasured.](https://flm.kuber.studio/research/colab-capacity.svg)

| Setting | Full 150M target | Full 300M target |
|---|---:|---:|
| Acquired neurons with computed fast/slow state | 166,700 | 166,700 |
| Modeled signed recurrent edges | 24,469,412 | 24,469,412 |
| Native BPE plus explicit special-token vocabulary | 4,101 | 4,101 |
| Learned lexical width | 728 | 1,600 |
| Calculated trainable parameters | 150,064,990 | 299,897,262 |
| FP32 parameters + gradients + two Adam moments, arithmetic only | 2.40 GB | 4.80 GB |

These memory figures exclude graph buffers, activations, temporary gradients, allocator reservation and saving. They are not peak VRAM measurements. The parameter formula is tested against instantiated small models. [Capacity record](https://github.com/Kuberwastaken/flm/blob/main/reports/colab-notebooks/capacity-plan-v1.json).

Both full presets retain the same neurons and routes. The additional capacity mostly enlarges learned input projections and lexical interfaces. **300M parameters does not mean twice as much biology.** The input projection drives every neuron artificially, pooling follows source order, and the rate/sign assumptions remain engineering choices. The acquired graph is not a physiologically complete animal. Zero-sign outgoing edges are omitted: 1,113,526 pairs, representing 3,917,219 contacts. The retained fast graph carries 120,260,398 / 124,177,617 acquired contacts, about 96.85%. This removes subset boundary cuts within the acquired graph; it does not restore unacquired connections or unknown physiology.

## A concrete memory improvement

The original [PyTorch 2.8 COO backward implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/autograd/FunctionsManual.cpp) forms a dense `grad.mm(mat2.T)` before applying the sparse mask. At 166,700 squared float32 entries, that intermediate alone is approximately **111.2 GB**. Scaling the previous COO training path directly would therefore fail even if parameter memory appeared modest.

The separate capacity runtime uses **CSR** and checkpointed unrolling, with an explicit chunked edge-gradient fallback. Small CPU checks compare loss and every parameter gradient against the original FLM; finite-difference checks cover the fallback. CSR support depends on the backend/build, so a full forward, backward and optimizer update is required before claiming T4 fit. The fallback supports first-order training, not higher-order derivatives. No measured speedup is claimed yet.

The full state and sparse routing remain float32; CUDA float16 autocast accelerates supported dense lexical operations. Eight-token checkpoint blocks recompute activations during backward while preserving gradients across the configured 128-token training window. There is no `detach` at those checkpoint boundaries. Windows themselves start from zero state, so this is not unlimited-context training.

## Run it

1. Open the notebook, select **T4 GPU**, and run cells in order. The notebook fetches an immutable FLM code revision and verifies graph/data hashes. Optional Google Drive mounting persists private run files across Colab sessions.
2. Keep `full-150m`, `flm`, and `pretrain` for the first attempt. The first three updates allocate optimizer state and emit actual throughput. Set `PROFILE_ONLY=True` for just this probe; otherwise it continues within the 40-minute trainer budget. An active update or save can overrun the soft deadline.
3. Inspect `report.json`: completed updates, scored/input tokens, validation snippets, peak allocated/reserved memory and measured runtime. An out-of-memory or nonfinite-gradient error is recorded as failure. The notebook does not quietly drop neurons or substitute a transformer.
4. To continue, keep the numerical settings and run name, and set `RESUME=True`. Model, Adam, scaler, RNG and cyclic data cursor are restored. Changed configuration, data hashes, source code or runtime identity are rejected. CPU resume parity is tested; bitwise GPU reproducibility across devices is not promised.
5. For conversation training, start a new `chat_sft` run and point `INITIAL_CHECKPOINT` at your own foundation checkpoint. It validates architecture, graph and tokenizer identity, retains the checkpoint lineage, and starts a fresh optimizer. Keep that phase distinct from pretraining.

`smoke` selects the existing 1,024-neuron graph at roughly 0.6M parameters. This checks the pipeline quickly. `transformer` selects a random Llama-style control with the same prepared corpus, tokenizer and label policy, approaching the same parameter target. It uses supported PyTorch SDPA, not the T4-incompatible settings in some upstream notebooks. Parameter counts and token exposures are reported; fixed wall time and fixed token budget answer different comparison questions.

## Data, output and acceptance

The pinned [source manifest](https://github.com/Kuberwastaken/flm/blob/main/data/sources/colab-v1.json) identifies **FineWeb-Edu** and **OASST1** revisions. Foundation preparation stops at 10,000 source documents or approximately four million training targets, whichever comes first. FineWeb-Edu text is filtered by score/length and normalized exact-document deduplication; classifier-selected web text does not guarantee human authorship. OASST1 requires reviewed English, non-synthetic, non-deleted parent chains and rank-0 assistant replies. Hash partitions keep complete trees together. Exact chat deduplication covers complete serialized branches; repeated ancestor messages and near-duplicates can remain.

The existing BabyLM-trained BPE supplies native tokens. Explicit BOS/EOS, role, end-of-turn and padding tokens create a new 4,101-ID vocabulary. This deliberately differs from the legacy browser ID-offset convention. No new tokenizer fitting uses validation/test. Conversation loss masks user/system text and role markers while learning assistant content and its end token. Windows do not cross documents or trees. Source cards, IDs, normalized hashes, token-array hashes and counts are saved locally. Licenses remain attached to each source; raw corpora are not committed or uploaded.

The two-window development panel catches broken learning cheaply. It is **not a representative held-out benchmark**. The notebook does not score its reserved test split, assert benchmark decontamination, or select favorable prompts. Its output cell records unedited generated tokens and actual fast state at each generated token. It saves loss/throughput plots and a state heatmap from those measurements.

An initial pass means finite complete updates with actual GPU memory/throughput recorded and usable checkpoint resume. It does not mean useful language. Continue a larger language budget only after a separate representative validation panel improves and unedited generations warrant it. Stop this configuration if it cannot complete the probe or its measured throughput makes the intended token exposure impractical. Do not turn failures into an open-ended anatomical subset search.

The frozen Mac selection study and [prospective anatomical gate](https://github.com/Kuberwastaken/flm/blob/main/docs/ANATOMICAL-PRIOR-DECISION.md) remain unchanged. A larger engineering model cannot reverse a valid negative anatomical result. Claims about wiring still require topology controls and effective-core controls under matched conditions.

## What is verified now

Eight focused CPU tests cover sparse/checkpointed gradient equivalence, finite differences, exact resume, changed-identity rejection, non-overlapping target windows, special-token separation, assistant-only masking, non-synthetic tree ancestry and the random transformer's training gradient. Short real-data integration runs exercise FLM pretraining, continuation onto OASST1 and the transformer control. [Local receipt](https://github.com/Kuberwastaken/flm/blob/main/reports/colab-notebooks/local-integration-v1.json). Their tiny budgets are plumbing checks, not quality results. The full graph was assembled and hash-verified locally; neither full-size preset has yet completed a T4 update.

Keep `last.pt`, its SHA256 receipt, run identity, report, tokenizer/data card and graph card together. Google Drive stores them privately. The notebook performs no Hub upload or model promotion. A 150M/full-graph checkpoint needs a separate browser export/runtime and quality review before it can replace the small models on GitHub Pages. The current live ChatFLM models remain available.
