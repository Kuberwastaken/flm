# Scheduling the frozen language study

The original language queue fits one condition at a time. A second scheduler
allows at most two independent training processes to use this CPU, each with
the same prescribed four PyTorch threads. The numerical trainer, graphs,
parameters, samples, optimizer, checkpoint choices and 6,000-update budgets are
bound to the existing study identity. The scheduler has its own recorded source
hash. This is an operational scheduling change after partial validation became
visible, not a new model or an additional performance-selected experiment.

At the resource check before this change, the original child used about 26% of
total CPU capacity and the machine had approximately 3 GB of available RAM,
with zero reported page reads. Its private allocation was about 1.16 GB and
working set about 327 MB. These are observations on a busy laptop; concurrency
is not assumed to improve throughput until both active jobs are measured.

## Run ownership and recovery

```powershell
python -m flm.language_topology_queue --workers 2 --score
```

The scheduler obtains an OS-held queue lease and checks live process commands
for existing language trainers before touching outputs. A stale lock file alone
does not block recovery. It verifies every completed reference and control,
resumes incomplete runs only from verified optimizer/RNG/exposure checkpoints,
and starts each remaining declared condition once in its original order. All
children use separate registered output directories. The additional worker
starts only when at least 2,048 MiB of physical memory is available; otherwise
the current process continues and the queue falls back to one worker.

Each child exit is followed by the existing complete-run verifier. A nonzero
child exit stops new launches while its live sibling finishes. A scheduler or
verification exception terminates only its owned live children; atomic saved
checkpoints remain resumable. If an external termination leaves an orphan child,
the next scheduler's live-process check rejects it. Inspect actual processes
before resuming; do not launch the legacy serial queue alongside this scheduler.
Use `--workers 1` to recover serial scheduling with the same verified runs.

With `--score`, the process invokes the existing all-ten-condition selection
gate only after every training child has completed and passed verification.
That gate freezes all selections before reading any new control test losses.
The dated public snapshot follows scoring. No separate finalizer should be
attached to this scheduler. Scientific review and browser validation still
precede committing and deploying completed test results.

## Transition from the original process

Before stopping the original queue, verify its latest checkpoint, including
sampler exposure, optimizer/RNG restoration, graph buffers and partial
validation selection. The 1,000-update checkpoint file is written after the
last/best checkpoint pair, providing a completed save boundary. Record the
checkpoint hash and saved step. Identify the actual old queue, its current
child and attached finalizer by PID and command. Stop the obsolete finalizer,
then the old parent and child, and confirm they are gone before starting the
new scheduler. It resumes that exact saved state. Work after the save boundary
may be replayed; it is not extra exposure in the resumed optimization history.

Training history files retain records from interrupted attempts. Their wall
times and throughputs are operational logs affected by contention. The
checkpoint-backed validation snapshot and audited sampler/RNG state determine
the scientific comparison; do not treat contended training logs as isolated
runtime benchmarks. All eight new conditions and both frozen measured
references still form the declared study.

## Verification

The scheduler tests cover the two-child concurrency bound, unique job launches,
completion of live siblings after a child failure, cleanup after failed
checkpoint verification, serial fallback under memory pressure, a real second
process blocked by the OS lease, lock release, exact command settings and
rejection of unverified resume state. The existing numerical/resume/selection
tests remain applicable because their implementations are unchanged.
