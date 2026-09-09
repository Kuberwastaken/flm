# Data attribution

## MaleCNS

Male central nervous system connectome, HHMI Janelia FlyEM and collaborators. [Official release](https://male-cns.janelia.org/download/). Licensed under Creative Commons Attribution 4.0 International; the license text is included in `CC-BY-4.0.txt`.

FLM's compact graph is a transformed subset: central-brain selection, source-ID reordering, modeled transmitter signs and normalized contact-derived weights. It is not an official physiological model. Source revision and transformations are recorded in the graph card.

The processed source distribution is [Neural Canvas](https://huggingface.co/spaces/Xenova/fruit-fly-simulation), revision `776d115ee5aa934578a87fd6d260d138084f59c1`. That distribution's code and individual assets retain their own licenses.

## LibriSpeech

Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. LibriSpeech: An ASR corpus based on public domain audio books, ICASSP 2015. [OpenSLR 12](https://www.openslr.org/12). CC BY 4.0.

FLM retrieves only transcript text and document IDs from the official `openslr/librispeech_asr` distribution, then lowercases, normalizes whitespace, groups utterances and removes specified exact duplicates. The data card records the transformations and partition limits. Training on transcripts does not train an acoustic model.
