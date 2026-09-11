# Language eligibility: complete window-gradient kernel

This is an implementation milestone for a future learning-rule study, not a
trained language model or evidence of an advantage over BPTT. The BabyLM
training queue and the completed language experiments are unchanged.

[`language_window_gradients`](../flm/language_eligibility.py) now combines the
[occurrence-based embedding traces](LANGUAGE-ELIGIBILITY-PREPARATION.md) with all
six recurrent/input trace groups and the instantaneous lexical classifier. It
returns a gradient for every FLM parameter, including both uses of a tied
embedding. Autograd differentiates only the current position's classifier;
recurrent temporal credit is computed explicitly using the declared local
approximation.

## Objective and update boundary

Inputs and targets are `[batch, time]` integer tensors. A separate boolean mask
chooses scored positions. The objective is the sum of their next-token cross
entropies divided by the number of scored positions across the entire window.
The caller must construct the desired mask, including warm-up and boundary-token
exclusions. There is no second averaging over batch, time or trace occurrences.

Every input advances the fast state, slow state and eligibility, including an
unscored warm-up input. An optional initial state supplies values only: gradients
do not cross the window's left boundary. Initial eligibility is zero. The default
forward state is also zero, matching the independent sampled training windows in
the current language protocol.

All weights remain fixed throughout a call. The function changes neither model
weights nor `.grad` fields, performs no clipping or optimizer step, consumes no
random numbers and returns detached tensors. A caller can assign the returned
gradients, clip once and perform one optimizer update after the complete window.
A new call reconstructs constants and starts fresh traces after that update.
Concurrent model mutation is unsupported; parameter version changes are checked
before returning.

The history-removal condition discards old eligibility at every position while
retaining exactly the same forward fast and slow states. It therefore tests
temporal credit assignment, not removal of forward memory. This is distinct from
the completed language study's recurrence-disabled or state-reset controls.

## Derivative and implementation boundaries

For each neuron, the retained fast temporal Jacobian is

    J_i,t = 1 - alpha_i + alpha_i (1 - z_i,t²) gain W_i,i.

The slow trace propagates through its own leak and the current fast trace.
Temporal paths between different neurons are omitted. Readout derivatives,
including pooling, normalization and the direct tied classifier derivative, are
computed exactly for the current numerical states. Contributions through the
input embedding are added separately; repeated input tokens accumulate into the
same embedding row.

The recurrent edge derivative includes incoming absolute-sum normalization, its
small-denominator floor, and the clamp on the edge log gain. These derivatives
matter when weights are clamped or a row has a very small total magnitude. The
existing frozen sensory implementation has not been edited.

The present kernel deliberately accepts only an ordinary full FLM or rewired
FLM, with dense computation and at most 2,048 neurons. It rejects altered
transition subclasses, sparse cores, frozen parameters and duplicate edge
entries rather than silently applying an unsupported derivative. A separately
specified fixed-core control is still required for the experiment.

Returned `persistent_trace_bytes` counts the explicit core traces and, when
enabled, the occurrence traces, token IDs and projection snapshot. It excludes
weights, optimizer state, accumulated gradients, cached input drives, retained
logits and temporary operations. This number is not peak memory or a timing
measurement. The occurrence contraction's cost still grows with sequence length.

## Numerical verification

The seven integration tests use tiny float64 fixtures, one CPU thread and no
corpus. They check:

- Every parameter gradient against ordinary BPTT in a self-only network, with
  tied and untied classifiers, unequal per-example masks and nonzero initial
  states.
- Every parameter against an independent autograd construction of the diagonal
  temporal approximation on a coupled graph. The same fixture exhibits nonzero
  differences from full BPTT in both embedding and edge gradients.
- History removal against an independently detached recurrence, with unchanged
  forward states and changed credit assignment.
- A token seen only in an unscored prefix receives an input-mediated gradient
  from a later loss when trace history is retained.
- One-position derivatives with sub-floor incoming magnitudes and clamped edge
  gains against ordinary autograd.
- Unchanged input model tensors, existing `.grad` fields and RNG during gradient
  computation; an AdamW checkpoint restored between two windows reproduces the
  uninterrupted second update exactly on the fixture.
- Invalid masks, token IDs, initial states and unsupported variants are rejected.

Run:

```powershell
python -m unittest discover -s tests -p test_language_eligibility.py -v
python -m unittest discover -s tests -p test_embedding_eligibility.py -v
```

These checks establish the declared kernel's numerical behavior. The independent
autograd reference intentionally removes cross-neuron temporal derivatives; its
agreement must not be described as agreement with full BPTT on coupled graphs.
Small-fixture checkpoint parity is not proof of complete trainer resumption or
bitwise reproducibility of future multi-thread training.

## Remaining work before fitting

Integrate a training-only runner with explicit data identities, exposure counts,
optimizer/RNG restoration, a matched fixed-core definition and all-condition
selection gates. Test its actual resumed windows, then measure disposable
full-size training-window costs after the priority queue finishes. Freeze the
dataset, compute budget, seeds, masking, selection and comparison rules before
fitting BPTT, fixed-core, eligibility and history-removal conditions. Keep failures
and inferior results. No full-size timing pilot, language eligibility fit or
held-out score has been produced by this implementation milestone.
