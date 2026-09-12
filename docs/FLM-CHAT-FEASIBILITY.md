# A larger conversational FLM: feasibility assessment

12 September 2026. **Proposal and planning estimates, not an implemented architecture, measured GPU throughput or new training queue.** This answers the request to assess a useful fly-derived chat model on the available laptops and a possible Colab T4. The frozen selection study and its continuation rule remain unchanged.

A useful model for a defined conversational scope is a reasonable research target. Broad parity with pretrained SmolLM-135M or SmolLM-360M is not a defensible near-term promise on this compute. Language quality, anatomical coverage and biological fidelity are different objectives; maximizing one does not establish the others.

## What the comparison actually requires

The current primary FLM has 600,003 trainable parameters, 1,024 scalar neuron states with fast/slow traces, and 76,130 directed edges. The default released BabyLM checkpoint received 18,432,000 training-token presentations. The 100M corpus designation describes its available word pool, not a claim that this checkpoint processed 100 million tokens or completed an epoch. See [architecture](ARCHITECTURE.md) and [BabyLM findings](BABYLM-FINDINGS.md).

The linked [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) and [SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M) are base models. Their model cards report 600 billion pretraining tokens and 64 H100 GPUs. The parameter counts are about 225 and 600 times FLM's current count; their token exposure is about 32,552 times greater. These ratios describe budgets, not a scaling law for FLM. [SmolLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) additionally receives conversation/instruction training and is the appropriate product comparator for chat.

Running or fine-tuning those existing weights is much cheaper than recreating their knowledge from random initialization. A transformer notebook is useful for data loading, checkpointing and comparison. Its pretrained weights would change the meaning of the FLM experiment if used to produce FLM's answers.

## What should remain from the fly

Retain identifiable neurons, measured directed routes, declared source-sign assumptions, circuit annotations and actual recurrent state. Route language output through that state. Audit what changes when recurrence is frozen or removed; merely connecting the core to the readout does not prove it contributes useful computation.

The acquired graph has 166,700 neurons and 25,582,938 directed pairs. Sparse storage of that graph is feasible in principle on a T4. With float32 weights and two int32 endpoint arrays, those three arrays alone require about 307 MB; two scalar float32 states per neuron require about 1.33 MB per sequence. This excludes optimizer state, graph buffers, lexical interfaces, batching and the many activations retained for gradients through time. Holding the graph is much easier than training it effectively over billions of tokens.

A reasonable **initial engineering envelope** is 8,000-16,000 computed neurons and roughly 10-30 million learned parameters, including the lexical interfaces. This is not a measured memory fit, optimal size or proposed search over subsets for favorable results. Select circuits by a documented anatomical rule, audit boundary cuts, and freeze that choice before quality comparisons. Prefer retaining annotated circuits and their measured connecting routes over selecting solely by degree. Report coverage of neurons, contacts and circuit membership separately.

Profile larger coverage, potentially the full acquired graph, before declaring a maximum. A frozen full-graph reservoir with a trained readout is a lower-training-cost alternative, but may give weaker language learning; it must be labeled fixed-core. More complex neuron dynamics or extra state channels also require explicit engineering assumptions. None restores missing biological physiology automatically.

The main implementation work is sparse GPU forward/backward computation, stable gradient propagation, cached graph structure, batched token processing and memory-efficient recurrent unrolling. The current dense path for small graphs and Python token loop cannot simply be enlarged and assumed fast. Graph partitions should preserve measured edges; padding blocks with new trainable connections would change the architectural claim.

Preservable engineering properties include fixed-size streaming state and neuron interventions. Those are shared with other recurrent models. Anatomical benefits in accuracy, sample efficiency, energy consumption or continual learning remain hypotheses. Local plasticity is a separate research direction, not an established shortcut to language competence.

## Data and actual conversation training

| Source | Proposed role | Important distinction |
|---|---|---|
| Existing BabyLM partitions | Preserve the low-data scientific comparison | Mixed speech/written text and transcript markup are not assistant instruction training |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Deterministic, deduplicated educational-text subset for a larger foundation | Web text selected using an educational classifier trained on Llama-generated ratings; not a guarantee that every web document is human-authored |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | English conversation branches, quality-filtered, non-synthetic records | Preserve conversation trees and roles; split by tree before extracting branches |
| [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Optional separately declared capability experiment | Includes synthetic Cosmopedia textbooks/stories; incompatible with the primary no-teacher-generated-text claim |

Start with a 20-100 million token learning pilot, then consider 0.5-2 billion tokens only if measured learning and throughput justify it. These are budget proposals, not data already acquired. Train the tokenizer on training documents only. Pin revisions, retain licenses and source hashes, deduplicate before splitting, check benchmark overlap, and record unique text separately from repeated token presentations. Dataset cleaning must create a new version; it must not silently alter the existing BabyLM benchmark.

Conversation fine-tuning needs explicit user/assistant boundaries, assistant-only prediction loss, end-of-turn targets and genuine multi-turn examples. It teaches how to use existing language capability; a few thousand conversations cannot substitute for pretraining. System prompts cannot create instruction following in a model that has not learned it.

Using synthetic training text can still mean weights trained from scratch. It cannot mean language learned independently of a teacher model. Keep any such variant separately named and documented. OASST1's card describes human-generated conversations, but the export schema also supports synthetic flags; enforce the desired provenance filter instead of relying only on the headline description.

## Compute and time

The inspected Windows host has an i5-1335U, about 16 GB RAM and integrated Iris Xe graphics. The Mac has an M5 Pro and 24 GiB unified memory. The Mac is useful for preparation, evaluation and CPU studies; large-core FLM training on its GPU has not been benchmarked. A [T4 has 16 GB device memory](https://www.nvidia.com/en-us/data-center/tesla-t4/), making it a useful development target. Arbitrary sparse recurrence does not automatically exploit its peak tensor-core throughput.

These are **hypothetical measured-throughput scenarios**, counting complete training updates rather than generation speed:

| Sustained training tokens/second | 100M tokens | 1B tokens |
|---:|---:|---:|
| 1,000 | 27.8 hours | 11.6 days |
| 5,000 | 5.6 hours | 2.3 days |
| 10,000 | 2.8 hours | 1.2 days |

They exclude engineering, comparisons, evaluation, interruptions and conversation fine-tuning. No row is a measured FLM GPU result. If throughput is 100 tokens/second, one billion tokens instead takes about 116 days: the early profile must be allowed to reject the proposed scale.

For scale only, assuming approximately six parameter operations per training token, repeating the original 600B-token budgets implies about 4.86e20 FLOPs for 135M parameters and 1.296e21 for 360M. Even dividing by a T4's advertised 65e12 FP16 operations/second gives approximately 87 and 231 continuous days. That is an optimistic dense-model arithmetic estimate, not an FLM speed prediction or proof that equivalent quality requires identical compute. Real training adds substantial inefficiency.

Colab offers no guaranteed GPU type or continuous allocation. Its [FAQ](https://research.google.com/colaboratory/faq.html) describes dynamic limits and generally at most 12-hour runtimes; eligible Pro+ execution can extend to 24 hours. A notebook must save model, optimizer, RNG and data-position state to persistent storage and verify resume behavior. Colab is not the permanent backend for a public chatbot.

I would allocate 1-3 days first to implementation profiling and a small learning check. If that succeeds, a first larger checkpoint is a 1-2 week planning target; a serious narrow-chat attempt with evaluation and browser integration deserves a 2-6 week budget. These are conditional work budgets, not guaranteed time to useful capability. Broad SmolLM parity has no evidence-based completion date here.

## What would count as progress

Define a limited task scope before training: for example, short explanations within a chosen domain and follow-ups that retain facts supplied earlier in the conversation. Reserve prompts and conversation trees that training never sees. Measure answer correctness, instruction following, context retention, repeated-output failures and latency. Inspect unedited generations with the UI repetition guard both documented and disabled for model-level scoring.

Use released SmolLM-Instruct as the practical capability reference, with its external pretraining budget disclosed. Train a small transformer and GRU on the same new corpus and budget for the scientific comparison. Retain rewired and fixed-core/recurrence-disabled comparisons when making anatomical or recurrent-mechanism claims. Budget-match and parameter-match comparisons answer different questions; report both when feasible.

Do not expand this into an open-ended search if held-out language learning stalls, the core is ineffective, or throughput makes the intended token budget impractical. Write explicit operational and quality acceptance thresholds before committing the larger run. The existing 128-fit study has its own fixed [stopping rule](ANATOMICAL-PRIOR-DECISION.md); a separate capability proposal cannot reverse its conclusion or justify further anatomical subset searches after a valid failure.

## Visuals and reactions

Render actual updated state every token, allow circuit selection and show interventions. The typing fly can remain visually responsive through authored motion. A larger checkpoint needs a measured browser inference path, probably a WebGPU implementation with CPU fallback; model download size and mobile behavior need evaluation before catalog promotion. GitHub Pages can continue serving the static client and model assets.

Learning to answer a question does not demonstrate changed attraction to sugar. To make that claim, compare frozen pre/post-training models under the same sensory inputs, motor interface and simulation seeds, and measure actual action differences. An animation reacting to text should be labeled illustrative unless a separately evaluated learned control policy drives it.

The useful target is an inspectable recurrent language model with a clearly defined conversational capability and measurable dependence on its core. Maximum neuron count and general chatbot quality should be reported as separate achievements.
