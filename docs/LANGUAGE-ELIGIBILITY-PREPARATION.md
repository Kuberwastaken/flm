# Embedding eligibility: a bounded language-learning primitive

This is preparation for an alternative learning-rule experiment. No language
model has been fitted with this implementation, and no language accuracy, speed
or memory advantage has been measured. The frozen language-control study and
the BabyLM queue are unchanged.

## Why the sensory implementation is insufficient

The completed [sensory study](LOCAL-LEARNING-PROTOCOL.md) replaces token embeddings
with four numeric inputs. Language adds a shared learned embedding, used both
before the recurrent core and in the tied output classifier. Its input-mediated
derivative can affect every neuron, even under the existing approximation that
omits temporal credit passing between different neurons.

A naive local embedding trace has indices batch, neuron, vocabulary entry and
embedding dimension, for both fast and slow state. With the current 1,024 neurons,
4,096 entries, width 96 and batch 16, float32 storage is **48 GiB**. Counting only
two traces per embedding parameter would miss the neuron dimension.

## Factorization within a window

Let A be the input projection, E the embedding, x_t the teacher-forced token,
d_i,t = alpha_i (1-z_i,t²), and J_i,t the retained diagonal temporal Jacobian
defined in the sensory protocol. Hold all weights fixed within the window and
start eligibility at zero. The local embedding derivative can be written

    e_h[i,v,k,t] = A[i,k] U_h[i,v,t]
    U_h[i,v,t] = J[i,t] U_h[i,v,t-1] + d[i,t] 1[x_t=v]
    U_s[i,v,t] = (1-beta[i]) U_s[i,v,t-1] + beta[i] U_h[i,v,t].

Linearity lets us instead retain one column for each input occurrence. Each new
column receives its direct term once; all existing columns continue through J
and beta. At a scored position, multiply the fast and slow columns by the
instantaneous readout signals, contract across neurons with A, and scatter-add
occurrence gradients into their token IDs. Repeated tokens contribute to the
same embedding row. The direct tied-output gradient must be **added** to this
input-mediated gradient, with the loss reduction already applied to the signals.

This factorization is exact relative to the stated diagonal temporal
approximation. It does not recover the omitted cross-neuron temporal paths and
is not generally equal to BPTT. Updating A inside the window invalidates the
simple factorization; the primitive snapshots A and rejects a changed projection
at gradient contraction. A future trainer must enforce the fixed-weight window
for all parameters, including the embedding, and reset traces at its boundary.

## Storage accounting

The following are explicit tensor representations, not process-memory peaks or
lower bounds on forward learning algorithms. Use `trace_accounting()` in
[the implementation](../flm/embedding_eligibility.py) to reproduce the counts.

| Representation, float32, batch 16 | Bytes | Binary units |
|---|---:|---:|
| Naive embedding traces, all vocabulary entries and dimensions | 51,539,607,552 | 48 GiB |
| Factorized traces, all vocabulary entries | 536,870,912 | 512 MiB |
| Occurrence traces, 96-position window | 12,582,912 | 12 MiB |
| Existing-style core/input traces, including shared gain per neuron | 22,851,840 | 21.79 MiB |
| Occurrence token IDs and projection snapshot | 405,504 | 0.387 MiB |
| Naive dense two-state Jacobian for all state-affecting parameters | 74,805,805,056 | 69.67 GiB |

The last row uses 570,723 state-affecting parameters: 177,507 core/input parameters
plus 393,216 embedding parameters. The other 29,280 classifier parameters do not
affect future states under externally supplied teacher-forced tokens. This
distinction would change for a differentiable feedback model.

The occurrence scheme uses about 34.18 MiB for these persistent traces and
metadata together with the existing core traces. It still needs model weights,
gradients, optimizer state, recurrent matrices and temporary contractions. Its
storage grows with window length. Contracting all active occurrences at every
loss can cost O(T²ND) arithmetic; avoiding a vocabulary-sized trace does not make
the method fast. BPTT may be cheaper in both measured memory and wall time.

## Checks and the next experiment

Four focused tests passed on tiny float64 fixtures:

- Occurrence traces match explicit dense local traces at every position,
  including repeated tokens and distinct token histories across the batch.
- Input-mediated plus direct tied-output embedding gradients match BPTT when
  recurrent connections are self-only.
- A coupled fixture exhibits a nonzero gradient discrepancy, preserving the
  approximation boundary rather than asserting false exactness.
- Invalid tokens, exhausted windows and changed projections are rejected;
  allocated trace sizes match the accounting.

Run `python -m unittest discover -s tests -p test_embedding_eligibility.py -v`.
These are numerical implementation checks, not language-learning results.

Before a real comparison, integrate the primitive with the six existing core
trace groups and instantaneous lexical gradients; check full-model state parity,
masked loss reduction, shared parameters, checkpoint restoration and update
boundaries. Then measure actual memory and time on disposable training-only
pilots. Declare a matched BPTT, fixed-core, eligibility and history-removal
protocol before fitting, using common data, initialization, exposure and
selection rules. Keep all failed and inferior results. Existing priority queues
finish first.

## Related work

[Bellec et al. (2020)](https://www.nature.com/articles/s41467-020-17236-y) develops
eligibility traces combined with neuron-specific learning signals. It motivates
the earlier sensory rule; this rate-model derivation does not reproduce its
spiking experiments. [Tallec and Ollivier's UORO](https://arxiv.org/abs/1702.05043)
uses a stochastic approximation for online recurrent gradients. The occurrence
factorization above is deterministic and retains a deliberately biased local
derivative; it is not UORO or a claim of a new general learning principle.
