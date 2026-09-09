# Compact comparison protocol

Registered before matched runs. This is a feasibility study, not a tuned leaderboard.

Compare FLM, its degree/sign-matched rewired control, recurrence-disabled and slow-state-disabled controls, a one-layer GRU and a two-layer causal transformer. Run seeds 42 and 43. All models use the same AMI training and validation files, byte vocabulary, sampled windows, 1,000 optimizer updates, batch 16, sequence length 96, 16 warmup positions omitted from the loss, AdamW learning rate 0.002, weight decay 0.01 and gradient clipping at 1.0. This matches presented training bytes (1,536,000 per run), not wall time or floating-point operations. The 6,000-update demonstration model is reported separately.

The GRU has embedding width 64 and hidden width 200. The transformer has two pre-normalized layers, width 104, four heads, a 208-wide GELU feedforward layer, rotary positional embeddings, no dropout, untied input/output embeddings and a causal 96-position attention window per layer. Their parameter counts are within 2% of FLM's 228,069. The controls preserve FLM's parameter allocation; disabling a mechanism leaves some parameters unused, which must be reported rather than presented as equal effective capacity.

Evaluate at 500 and 1,000 updates on fixed validation document prefixes (32,768 scored bytes total). Select the lower-validation-loss checkpoint within each run. Reset state per document and carry it between chunks. The transformer retains a bounded per-layer key/value cache; recurrence carries compressed state. Both inference paths must match whole-sequence causal evaluation under chunking. The effective accessible history differs by architecture and is part of the comparison, not an equal-context guarantee.

Do not inspect test loss until this protocol and the runs are fixed. Test evaluation uses the complete 16 held-out documents. Report byte-weighted loss, per-document loss, seed variation and a paired document bootstrap for differences. Bootstrap intervals concern these documents and do not eliminate uncertainty over participants, domains, hyperparameters or random seeds. Two seeds provide only a limited estimate of training variability.

Include a smoothed character n-gram reference fitted only to training text, with smoothing chosen on validation. It is not parameter-matched. A graph advantage is supported only if the measured graph consistently beats its randomized control under this protocol. Failure is a result worth publishing. No result from this setup warrants a claim that fly wiring beats modern large transformers in general.

Implementation references: [PyTorch 2.8 scaled dot-product attention](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html); [RoFormer / rotary positional embeddings](https://arxiv.org/abs/2104.09864). FLM's core equations and source assumptions are in `ARCHITECTURE.md`.
