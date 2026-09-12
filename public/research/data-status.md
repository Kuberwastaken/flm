# Data and acquisition status

Updated 12 September 2026. The browser default is **BabyLM 100M FLM, seed 43**. The catalog contains 18 completed primary WikiText/BabyLM checkpoints plus legacy AMI. Training progress for the new selection cohort is separate from these completed models.

## Completed primary datasets

- **BabyLM 2026:** all 24 official files at pinned revisions are acquired, with 706,177,689 verified source bytes. The 10M/100M training pools share validation/test partitions. The train-10M-fitted 4,096-ID tokenizer, source-indexed caches and overlap audits are complete. All twelve baseline fits, frozen selections, held-out comparisons and 288 continuations are published in [the findings](https://github.com/Kuberwastaken/flm/blob/main/docs/BABYLM-FINDINGS.md). The 128-fit selection study reuses the pinned 10M inputs at a smaller exposure budget; its own validation and held-out comparisons remain pending.
- **WikiText-2 raw:** all six baseline fits and their held-out evaluations are complete, using 600/60/60 article partitions and a separate train-only 4,096-ID tokenizer. The topology and retrained computation comparisons are also complete. See [the protocol](https://github.com/Kuberwastaken/flm/blob/main/docs/WIKITEXT-PROTOCOL.md) and [computation results](https://github.com/Kuberwastaken/flm/blob/main/docs/LANGUAGE-CORE-RESULTS.md).

The machine-readable cards in `data/cards/` and `data/tokenizers/` retain exact sources, transformations and counts. A 100M-word training pool is not 100M words consumed: every completed BabyLM baseline fit received 18,432,000 token presentations sampled with replacement. FLM does not claim an official BabyLM challenge ranking; it reports its declared likelihood and generation panels.

## Earlier and deferred datasets

AMI meeting transcripts remain a separate dialogue experiment, acquired from the
official archive, hashed, normalized and divided into participant-disjoint train,
validation and test partitions. See `data/cards/ami.json` for the source checksum
and custom split rules.

LibriSpeech is an optional follow-up corpus. The dataset viewer returned HTTP 429 during acquisition. Verified batches remain in the ignored local cache; the download is incomplete and no partial LibriSpeech corpus has been used for training or evaluation. The downloader now stops on rate limiting and cancels queued work. Its default concurrency is one.

The trained release uses orthographic text, not audio, pretrained embeddings or language-model-generated training examples. Speaker turns are linearized; overlap, prosody and many aspects of natural interaction are therefore absent.

SCAN's official simple, length and added-primitive-jump partitions are acquired at a pinned publisher revision and checked against all 20,910 canonical commands. Its artificial instruction/action pairs remain separate from the primary language corpus. No official model has been fitted or scored on SCAN; see [the data audit and design](https://github.com/Kuberwastaken/flm/blob/main/docs/INSTRUCTION-TRANSFER.md). SCAN, LibriSpeech completion, broader scaling and alternative-learning sweeps are deferred under the [current priority](https://github.com/Kuberwastaken/flm/blob/main/docs/PLAN.md), not queued behind training.
