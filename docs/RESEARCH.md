# Research basis

The core question is whether a declared fly-derived wiring constraint helps sequence prediction when data is scarce. Biological structure is an inherited prior; training from scratch does not mean learning without priors.

## Evidence

- [Google Research, male fruit-fly connectome announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) and [Janelia's MaleCNS release](https://male-cns.janelia.org/download/): anatomical connectivity, not a recovered pretrained intelligence. Data is CC BY 4.0.
- [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9): simplified connectome-based dynamics can predict specified sensorimotor responses. This does not validate language learning or every MaleCNS transfer.
- [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3): task optimization of connectome-constrained networks provides a precedent for learning within anatomical structure.
- [Morra et al., 2023](https://arxiv.org/abs/2306.01885): fly-derived reservoir computing is prior art; the use of a connectome alone is not a novelty claim.
- [Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y): eligibility traces need appropriately defined learning signals. Simple Hebbian updates are not equivalent to next-token gradients.
- [SpikeGPT](https://arxiv.org/abs/2302.13939), [RWKV](https://arxiv.org/abs/2305.13048), and [Mamba](https://arxiv.org/abs/2312.00752): alternative recurrent language models are established comparison points. Spikes do not automatically save energy on ordinary hardware.

## Available local evidence

The acquired graph retains 166,700 annotated neurons, 25,582,938 directed neuron-pair edges and 124,177,617 contacts. Its source snapshot is [Neural Canvas revision 776d115ee5aa934578a87fd6d260d138084f59c1](https://huggingface.co/spaces/Xenova/fruit-fly-simulation/tree/776d115ee5aa934578a87fd6d260d138084f59c1). All 127 original source/asset files were hash-verified. Source count and orientation checks passed; the independent CPU stimulation/disconnection assay demonstrated propagation through the graph.

The source model card disagrees with runtime metadata for dopamine, serotonin and octopamine signs. FLM must explicitly version its modeling rule rather than silently mix them. Source contact counts are not measured functional strengths. A visual soma position is not a full neuron arbor.

The full reference body comprises 39 NeuroMechFly STL meshes. The body specimen is female and the graph is male; any visual combination must retain that distinction. The language core does not require a physical body.

## Dataset decisions

- [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext): the main matched comparison now uses the official 600/60/60 article partitions, a train-only 4,096-token vocabulary and near-600k-parameter FLM/GRU/transformer models. See `WIKITEXT-PROTOCOL.md`.
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/): the initial compact experiment uses CC BY 4.0 manual meeting transcripts. The recorded custom split separates globally identified participants and related meeting families; see `data/cards/ami.json`. The primary training partition contains 139 documents and 947,966 whitespace-delimited words including turn labels.
- [LibriSpeech, OpenSLR 12](https://www.openslr.org/12): CC BY 4.0, recorded English audiobook speech with transcripts. Optional acquisition stopped after service rate limiting. No incomplete partition was used. See `DATA-STATUS.md`.
- [Official Hugging Face distribution](https://huggingface.co/datasets/openslr/librispeech_asr): possible text-only retrieval via the dataset viewer, avoiding large audio downloads. Record the revision and exact retrieved rows because the viewer endpoint may track the current dataset.
- [BabyLM guidelines](https://babylm.github.io/guidelines.html): useful low-data evaluation framing, but corpus components mix speech and written text and require individual provenance checks.

No scraped private speech, pretrained embeddings, language-model-generated corpus or teacher-model outputs enter the primary experiment. Any later dataset or frontend that changes this rule gets a separate model/data card.

## Evaluation requirements

Measure held-out byte cross-entropy, perplexity only within a fixed tokenizer, train/validation gaps, throughput and memory. Compare against unigram/n-gram, recurrence-disabled, randomized-wiring and conventional recurrent baselines as compute permits. Use identical splits, vocabularies, context handling and tuning budgets. A label permutation is not a topology randomization.

Separate persistent training weights from episode-local adaptation. Use only past observations for prediction; reset state and adaptive memory at declared boundaries. Do not infer language understanding from a rendered bright region or a few attractive samples.

Sources accessed 10 September 2026. Model design and measured experiment reports are maintained separately so proposals cannot be mistaken for results.
