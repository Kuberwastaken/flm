# Data attribution

## MaleCNS

Male central nervous system connectome, HHMI Janelia FlyEM and collaborators. [Official release](https://male-cns.janelia.org/download/). Licensed under Creative Commons Attribution 4.0 International; the license text is included in `CC-BY-4.0.txt`.

FLM's compact graph is a transformed subset: central-brain selection, source-ID reordering, modeled transmitter signs and normalized contact-derived weights. It is not an official physiological model. Source revision and transformations are recorded in the graph card.

The processed source distribution is [Neural Canvas](https://huggingface.co/spaces/Xenova/fruit-fly-simulation), revision `776d115ee5aa934578a87fd6d260d138084f59c1`. That distribution's code and individual assets retain their own licenses.

## LibriSpeech

Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. LibriSpeech: An ASR corpus based on public domain audio books, ICASSP 2015. [OpenSLR 12](https://www.openslr.org/12). CC BY 4.0.

FLM retrieves only transcript text and document IDs from the official `openslr/librispeech_asr` distribution, then lowercases, normalizes whitespace, groups utterances and removes specified exact duplicates. The data card records the transformations and partition limits. Training on transcripts does not train an acoustic model.

## AMI Meeting Corpus

The AMI Project and corpus contributors. [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), manual annotations version 1.6.2. The official release licenses signals and transcription under CC BY 4.0. See the [download and license page](https://groups.inf.ed.ac.uk/ami/download/).

FLM uses word/punctuation transcriptions, omits nonlexical events, linearizes overlapping segments, applies generic speaker labels and lowercases the text. Its custom split keeps participants and meeting families together. No audio or unrelated annotation types are used. Archive hash, transformations, assignment rules and limitations are in `data/cards/ami.json`.

## SCAN

Brenden Lake and Marco Baroni, *Generalization without Systematicity*, ICML 2018.
The [publisher repository](https://github.com/brendenlake/SCAN/tree/c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f)
contains a BSD license notice for CommAI-env software, copyright Facebook,
2016-present. The exact source notice is retained in `SCAN-BSD.txt`; GitHub's
license metadata reports `NOASSERTION`. `data/cards/scan.json` preserves this
distinction and the source hashes. FLM audits the complete command universe and
three official partitions, preserving duplicate training rows. Its action codec
is an explicit derived representation for a future instruction-transfer study.
No neural-model result or physical motor behavior is implied by data preparation.
