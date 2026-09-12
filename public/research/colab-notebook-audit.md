# Colab small-model notebook audit

12 September 2026. Code and saved outputs inspected at immutable Git revisions. These are upstream records, not FLM GPU measurements. The [machine-readable audit](https://github.com/Kuberwastaken/flm/blob/main/reports/colab-notebooks/source-audit-v1.json) records paths, revisions, hashes and licenses.

**150Ã¢â‚¬â€œ300M parameters is a reasonable T4 engineering target.** The earlier feasibility note placed too much emphasis on reproducing SmolLM's original 600B-token budget. That is neither a minimum data requirement nor a reason to exclude a short, larger FLM experiment. Memory fit, training throughput and useful conversation need separate measurements.

| Notebook | What actually trains | Saved evidence | What transfers to FLM |
|---|---|---|---|
| [Unsloth Gemma 3 270M](https://github.com/unslothai/notebooks/blob/bce1b9d12e566c9df217fa236904ae3da3415247/nb/Gemma3_%28270M%29.ipynb) | Pretrained instruction model; rank-128 LoRA on chess examples | **Tesla T4, 100 updates, 30,375,936 trainable adapter parameters, 497.53 seconds / 8.29 minutes** | Response-only labels, short budget, memory/time reporting; strongest directly timed T4 hit |
| [Hugging Face SmolLM2-135M SFT](https://github.com/huggingface/smol-course/blob/32dde01a1110fed92a03cc16316c868645a35e9f/v1/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb) | Pretrained base model; rank-6 LoRA; conversation data | Saved output: 72 updates in 195.24 seconds; GPU unspecified; packing changes the number of training sequences | Explicit roles, packing and accumulation. Do not infer document exposure from the packed step count or call this a verified T4 timing |
| [Google Gemma 2 2B cookbook](https://github.com/google-gemini/gemma-cookbook/blob/3c2935f537ecb667ad8444490e13f1edcfef993c/Gemma/%5BGemma_2%5DFinetune_with_Unsloth.ipynb) | Pretrained, 4-bit base; rank-16 LoRA | Saved T4 identification; 60 updates configured; 20,766,720 trainable parameters | Checkpoint workflow and low-memory adaptation |
| [Google Gemma 3 270M emoji notebook](https://github.com/google-gemini/gemma-cookbook/blob/3c2935f537ecb667ad8444490e13f1edcfef993c/Demos/Emoji-Gemma-on-Web/resources/Fine_tune_Gemma_3_270M_for_emoji_generation.ipynb) | Pretrained model, LoRA plus lexical modules; three epochs | No saved runtime; includes bf16 assumptions and a local CSV step | Narrow task definition, before/after generation and export |
| [Hugging Face SmolLM-135M GRPO](https://github.com/huggingface/notebooks/blob/main/course/en/chapter13/grpo_finetune.ipynb) | Pretrained instruction model; rank-16 LoRA; length reward | No saved runtime; rewards proximity to **50 characters**, not answer correctness | Useful later as a reward-learning reference, not initial language acquisition |

Trainer time excludes installation, downloads, preprocessing and export. None of these five notebooks documents training a useful 150Ã¢â‚¬â€œ300M model from random weights in an hour. That does not prevent running a meaningful learning pilot in an hour.

The SmolLM GRPO notebook's FlashAttention 2/bfloat16 settings need changing for a T4. [FlashAttention's supported hardware](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support) excludes Turing from its standard FA2 path. Use float16/autocast and supported attention for the comparator; FLM has no attention layer. Unsloth's transformer patches do not accelerate arbitrary sparse recurrence automatically.

## A more relevant from-scratch precedent

[L20-Edu-135M](https://arxiv.org/html/2606.22189v1) reports a randomly initialized 134.5M transformer trained on one **48GB L20**, with 10B initial tokens taking about **72 GPU-hours**, followed by 3B continued-pretraining tokens. Its self-run six-task mean is 0.4150 versus SmolLM-135M's 0.4767. This is useful evidence that substantially smaller budgets can produce meaningful small-model capability. It is a single-run, self-reported preprint; some checkpoint selection uses the reported benchmark, and its L20 throughput is not a T4 or FLM measurement.

## Adaptation direction

The FLM notebook follows the successful short-run workflow but implements its own recurrent trainer: random initialization, explicit graph coverage, float32 sparse state computation with mixed-precision lexical matrices, checkpointed unrolling, response-only conversation loss and resumable timed runs. No upstream notebook code is vendored or relicensed. Its source inspirations retain the licenses recorded in the audit; FLM's implementation remains under the repository license.

The first deliverable is a resource and learning probe, with 150M/300M target presets and actual parameter-group reporting. Growing lexical projections is not equivalent to adding biological neurons. No checkpoint from this work may be described as SmolLM-equivalent, more biological, or a better chatbot until the corresponding measurements exist. This separate engineering experiment does not alter the frozen selection study or its anatomical continuation gate.
