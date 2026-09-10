# Acquisition status

The browser's primary language release uses WikiText-2 raw with 600/60/60 articles
and a frozen train-only 4,096-ID vocabulary. All official source files, article
identities and split caches are verified. See `data/cards/wikitext2.json` and
`data/tokenizers/wikitext2-4096/` for the exact transformation and counts.

The next study has acquired all 24 official BabyLM 2026 files at pinned revisions:
10M and 100M training words plus common validation/test partitions. Acquisition
verified 706,177,689 source bytes. Completed substantial-line overlap measurements
are in `data/cards/babylm-2026-overlap.json`. The lossless 10M-fitted vocabulary,
source-indexed memory-mapped caches and fixed validation panel are prepared with
`flm.babylm_prepare`; its final tokenization card records completed partitions.
See `docs/BABYLM-PROTOCOL.md` before interpreting model results or split integrity.

AMI meeting transcripts remain a separate dialogue experiment, acquired from the
official archive, hashed, normalized and divided into participant-disjoint train,
validation and test partitions. See `data/cards/ami.json` for the source checksum
and custom split rules.

LibriSpeech is an optional follow-up corpus. The dataset viewer returned HTTP 429 during acquisition. Verified batches remain in the ignored local cache; the download is incomplete and no partial LibriSpeech corpus has been used for training or evaluation. The downloader now stops on rate limiting and cancels queued work. Its default concurrency is one.

The trained release uses orthographic text, not audio, pretrained embeddings or language-model-generated training examples. Speaker turns are linearized; overlap, prosody and many aspects of natural interaction are therefore absent.
