# Exploratory mechanism diagnostics

Declared after the first WikiText training seed completed and before running these
diagnostics. The main six-run language comparison remains unchanged. These are
validation diagnostics; they do not inspect the held-out test split or select new
training settings for the registered models.

For each completed FLM seed, restore its validation-selected checkpoint and score
the same 32,768-position validation-prefix protocol in four conditions:

1. Reconstruct the original untrained model with that run's initialization seed.
2. Use the trained model without intervention.
3. Remove inter-neuron recurrent input in the trained model, retaining its learned
   fast leak, slow filter, input and readout.
4. Zero the trained model's slow state, retaining its recurrent fast dynamics.

The intervention conditions receive no additional optimization. They therefore
test sensitivity of a trained implementation to removing a mechanism. They are
not retrained ablations, comparisons of learning efficiency, or evidence that
anatomical wiring outperforms a randomized graph. The recurrence-off condition
still has temporal memory through its leaky fast and slow states.

Recompute the untrained and trained intact scores and compare them to their
recorded validation artifacts as a reconstruction check. Publish all four results,
including changes that improve loss. Record checkpoint, tokenizer, graph and
validation-cache hashes, exact evaluated bytes, and per-article losses.

Also replay one original paragraph from zero state through the untrained and
trained models. Keep the exact input identical. Record tokenwise mean absolute
fast/slow activity and RMS distance between the paired state vectors. This
describes a change in the artificial model's input response. It does not measure
knowledge, animal experience, semantic localization or learned motor behavior.

Future causal architectural comparisons require training the degree/sign-matched
rewired, recurrence-off and slow-state-off models from scratch under the same
registered schedule, across multiple seeds. They are separate follow-up studies.
