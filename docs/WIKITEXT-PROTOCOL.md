# WikiText-2 lexical study protocol

Registered before the first run. This replaces AMI as the primary comparison; the earlier byte study and its data remain separate.

Use the official WikiText-2 raw partitions, exactly preserved and grouped into 600/60/60 articles. The 4,096-ID lossless byte-BPE tokenizer is learned only from training text. All models use this same tokenizer and the same independently seeded sampling stream. No pretrained model, tokenizer or synthetic teacher data is used.

| Model | Trainable parameters | Core |
|---|---:|---|
| FLM lexical | 600,003 | 1,024 selected neurons, 76,130 signed edges, 128 pools, fast/slow state |
| GRU | 595,408 | One recurrent layer, hidden width 200 |
| Transformer | 607,468 | Two pre-norm layers, width 108, six heads, feedforward width 216, RoPE, 96-position causal window per layer |

Every model has a 96-dimensional tied input/output lexical embedding. FLM projects its 256 normalized pooled features to this lexical space. The GRU and transformer project their respective states to the same space. The FLM topology and two timescales remain unchanged; this is a learned language interface, not added biological anatomy. Parameter counts differ by less than 2%.

Train each model for 6,000 updates, batch 16, 96 tokens per sequence, excluding the first 16 positions from the loss. This presents 9,216,000 input tokens per run (about three passes over the 3,083,090-token training partition, sampled with replacement). Record exact decoded bytes presented and scored; do not equate token counts with word counts. Use AdamW, weight decay 0.01, gradient clipping at 1.0, 100-step learning-rate warmup to 0.002, then cosine decay to 0.0002. Lexical embeddings initialize with standard deviation 0.2; readout projections with 0.025. Core-specific initializations are documented in code. No dropout is used. Use seeds 42 and 43 for the matched comparison.

Evaluate every 500 updates on deterministic per-article prefixes totaling at most 32,768 scored token targets; choose the best checkpoint on validation bits per UTF-8 byte. Save numbered checkpoints every 1,000 updates for exposure curves and fixed-prompt samples. Evaluation resets at article boundaries, carries native state within an article, and excludes the two inserted boundary IDs from likelihood. Token byte lengths reconstruct the exact denominator. Report both shared-tokenizer perplexity and bits/byte, without comparing them directly to published word-token WikiText perplexities.

After runs and selection are fixed, score all test articles, report per-article losses and seed variation, and bootstrap paired article differences. The n-gram reference is fitted on the full training text with smoothing selected only on validation. It is not parameter-matched. Add the rewired and disabled-mechanism controls after the main FLM/transformer comparison, retaining the same protocol if making topology claims. Do not infer a graph advantage from the main architecture comparison alone.

Quality review uses a fixed set of original prompts and fixed sampling settings at matched checkpoints. Include failures, repetition, unsupported claims and state sensitivity. Sampling examples are illustrative, not evidence of factual accuracy. Brain, memory and body studies remain separate from language-quality claims.
