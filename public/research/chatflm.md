# ChatFLM: language through fly-derived wiring

FLM is trained from scratch on top of a connectome-derived recurrent core. **No pretrained or overfitted transformer supplies FLM's answers or its displayed neural activity.** Learned text embeddings drive the selected fly graph; its fast and slow state produces next-token probabilities through a learned readout. The brain is a 1,024-neuron subset, not an intact fly brain. “From scratch” describes the training origin, not general foundation-model capability.

The useful feature is inspectability: compare predictions, replay an input with a neuron silenced, inspect fast versus slow state, and measure a separate local readout update. [Mechanism controls](language-core-findings.md), [training-induced dynamics](language-dynamics-findings.md) and [physical reference experiments](food-core-physical-results.md) answer different questions. The visible typing motion is a designed illustration using the existing articulated body geometry.

## Models and the automatic default

The selector contains all **18 completed primary neural checkpoints**: FLM, GRU and transformer, each with seeds 42/43, on WikiText-2 and BabyLM 10M/100M. The earlier AMI model is also available. The separately trained GRU and transformer are comparison architectures; selecting them does not create anatomical neuron activity. The topology and computation-control models remain available in their separately documented inference bundles.

Opening ChatFLM without a model query selects **BabyLM 100M FLM, seed 43**, the lowest fixed-panel validation BPB among the four completed BabyLM FLM fits (**1.931862680356**). These candidates share validation text and tokenizer. Test loss and attractive continuations are not selection criteria; WikiText losses are not mixed into this ranking. This selects the strongest measured FLM candidate on that panel, not the strongest architecture overall. Explicit model links are respected, including the older `?model=babylm` and `?model=wikitext` releases.

The [catalog](/models/catalog.json) records checkpoint manifest hashes and selection-source hashes. Each package has its own weights, tokenizer and Python parity fixture. The browser verifies manifest, weight and tokenizer hashes before inference. [Catalog tests](/research/browser-catalog-test.js) compare all 4,096 logits at prefix lengths 1, 12, 97 and 130 for every model, crossing the transformer's 96-position window; FLM's fast/slow state is checked too. These are numerical correctness checks, not a performance benchmark.

## Conversations and system context

Chat mode stores user/assistant turns and an optional system prompt. **The checkpoints are base next-token models without instruction tuning.** The interface supplies a real conversation history; it does not add a hidden assistant or guarantee that the model follows roles, facts or requests.

The exact input uses ordinary text:

```text
System: optional context

User: first message
Assistant: previous generated reply
User: latest message
Assistant:
```

Open **Chat settings → Inspect the model's input** to inspect the pending input. Exported conversations also retain each reply's exact input, checkpoint hash and sampling settings. The 16,000-byte input limit drops oldest complete turns when needed, preserves system context and the latest user turn, and reports how many older messages were omitted. This is an input-size bound, not a guarantee of memory retention: FLM compresses history into recurrent state, while the transformer uses a finite attention cache.

Generation stops at EOS, the token cap, Stop, or a new line beginning `User:`, `Assistant:` or `System:`. Role stopping is display logic and can also truncate a quoted role label. The complete decoded sampled suffix is retained as `rawText` in the export; displayed text and raw text can differ. Nothing is rewritten into a better answer. Token/byte counts describe sampled output, including any stopped marker. **Continue text** preserves literal continuation behavior; **Legacy dialogue** retains the earlier AMI speaker format.

Enter sends; Shift+Enter inserts a line. History, system context and exported learning stay scoped to the selected checkpoint. Browser storage may be unavailable or full; exports remain available. Local adaptation updates a separate readout adapter, not the released recurrent weights. Switching models reloads the workspace, so save or export local learning before switching.

## Workspace and scope

Desktop has conversation on the left, actual FLM state at upper right and the 3D typing fly below it. Neuron controls, predictions and the original state-to-pose body comparison are under disclosures. Small screens stack these views. Light/dark mode follows the system until manually selected, then persists locally. Reduced-motion preferences suppress typing movement.

The layout draws on common conversation, composer and model-selection patterns documented in [AI Elements](https://elements.ai-sdk.dev/examples/chatbot). The implementation uses native HTML controls and the existing local worker; it does not import that template's framework or hosted model service.

See the [completed BabyLM findings](babylm-findings.md) for comparative losses and output limitations, and the [registered selection study](selection-language-protocol.md) for the continuing test of which circuits to retain. A chat interface is not evidence of reliable chatbot performance.
