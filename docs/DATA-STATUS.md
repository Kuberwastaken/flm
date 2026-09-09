# Acquisition status

The primary experiment uses AMI meeting transcripts, acquired from the official archive, hashed, normalized and divided into participant-disjoint train, validation and test partitions. See `data/cards/ami.json` for assignments, counts, limitations and the exact source checksum.

LibriSpeech is an optional follow-up corpus. The dataset viewer returned HTTP 429 during acquisition. Verified batches remain in the ignored local cache; the download is incomplete and no partial LibriSpeech corpus has been used for training or evaluation. The downloader now stops on rate limiting and cancels queued work. Its default concurrency is one.

The trained release uses orthographic text, not audio, pretrained embeddings or language-model-generated training examples. Speaker turns are linearized; overlap, prosody and many aspects of natural interaction are therefore absent.
