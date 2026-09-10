# Browser validation log

Local desktop browser, checkpoint 3,000, 10 September 2026. These are functional checks, not held-out release benchmarks.

- Browser generation produced 400 bytes from the actual worker, with changing neural state and next-byte probabilities. An earlier 1,000-step checkpoint produced approximately 105 bytes/s including the deliberately slowed observation playback; this is a single interactive observation, not a standardized throughput benchmark.
- Anatomy and checkpoint checksums verified before use. Both WebGL views rendered without captured console errors. The brain framing was corrected to exclude the VNC from the default anatomical context crop; all 1,024 model neurons remain in inference. Five selected neurons have no soma coordinate and remain inspectable through the selector.
- The articulated body loaded all 39 mesh files. Automated FK tests match all reference joint coordinates. Body pose is explicitly illustrative.
- Local adaptation test: two passes over 139 UTF-8 bytes of original test dialogue, rate 0.03; a separate 50-byte probe improved from 3.737 to 2.674 bits/byte. Online training loss was 2.718 bits/byte. This small, related probe is not evidence of broad generalization. The UI reported 278 completed updates and successfully saved the adapter in browser storage.
- At the 390-pixel responsive viewport setting, the browser's document client width and scroll width were both 375 pixels: no horizontal document overflow. Desktop layout places conversation on the left and brain/body on the right; narrow layout stacks them.

Outstanding release checks include production-bundle behavior, cancellation, intervention replay, storage roundtrip, keyboard navigation, failed asset handling, download links and the deployed custom domain. Add verified results here as those checks complete.


## WikiText browser revision

Production preview, 10 September 2026: the 1,000-update lexical checkpoint generated 200 actual tokens / 751 bytes from the original prompt `The history of science`, at 73.7 tokens/s including observation delay. It produced sentence fragments, invented details and repetition. This is a functional observation, not a quality or throughput benchmark. Both 3D canvases loaded, all 1,024 model identities were available, and no browser console errors were captured.

The Research view loaded three validation curves, selected the common 1,000-update point, and displayed FLM 2.137, GRU 2.127 and transformer 2.114 bits/byte. Switching the fixed prompt updated all three unedited sample columns. Full-page capture showed stitching overlaps; a DOM audit confirmed unique IDs, one source list and one instance of each of the eight research headings.

Automated worker integration verifies generation, scoring, adaptation, clearing, neuron silencing and cancellation during priming for both AMI and WikiText. The 27 Python checks and 26 JavaScript checks passed after lexical support. Binary browser/PyTorch parity also passed for the subsequently exported 3,000-update WikiText checkpoint. The later packed-adapter storage format still needs a direct browser save/reload check.


## Live WikiText verification

The 3,000-update WikiText package is deployed at the custom domain with HTTPS enforcement. The Pages workflow succeeded at commit `8cf2ab2`. The live worker learned two passes / 62 token updates from 103 bytes of original garden-and-bird text: online training 2.447 bits/byte, separate 47-byte probe 3.983 to 3.800 bits/byte. Saving the packed adapter succeeded; after reload, Load saved reported that the checkpoint-specific learning was restored. This is a functional related-text check, not a benchmark result.

At the 390-pixel viewport setting, the live Research view had document client/scroll widths of 375/375 pixels. Sample columns stacked into a 335.2-pixel single column. All three measured comparison rows were visible and no console errors were captured.

## Completed-checkpoint and extended research checks

Production preview, WikiText checkpoint 6,000: keyboard generation produced 24
actual tokens. Switching between AMI and WikiText, selecting an existing
conversation from the other dataset, and importing an AMI archive while WikiText
was selected preserved the correct model/conversation association. An incompatible
learning file was rejected without disabling the loaded model. Export exposed the
complete 2,119,150-byte adapter JSON and Copy JSON succeeded. Native download events
were unavailable in this in-app browser, so file delivery is not inferred from a
button click; the explicit JSON fallback is verified.

The extended Research page displays all six measured BabyLM component counts.
Neural-response and physical-trajectory panels load correctly and stack vertically
at a 390-pixel viewport without document overflow. The four-second physics video
played to completion (duration/currentTime 4, ended true, readyState 4, no media
error). Its recording represents one simulated second at quarter-speed playback.
The viewport override was reset after verification. The complete verification
suite now passes 35 Python tests and 27 JavaScript tests, including identical
sampling/scoring between uint16 memory-mapped and int64 in-memory token data.

The completed-study production build shows all six test scores, all four paired
article intervals, the separately tuned n-gram and the isolated CPU/state table.
Displayed values agree with the numerical artifacts. The fixed 6,700-pair grammar
panel shows both seeds and their means; mobile result tables remain inside their
horizontal scroll containers without document overflow. Final verification after
the linguistic scorer addition passes 37 Python and 27 JavaScript tests. The
updated eight-page methods report and five-page data note were rendered and all
13 pages inspected; the final bibliography change was re-rendered and reviewed.

## BabyLM evaluation and progress interface

The complete evaluation path adds nine meaningful Python checks: ragged batched
likelihoods versus independent blocks for FLM/GRU/Transformer; overlap exclusion;
paired block intervals; incomplete-study gating; changed-budget/checkpoint
rejection; cached-batch identity/count validation; reproducible sampling, EOS,
control-byte filtering, invalid UTF-8 and repetition accounting. The full suite
passes 46 Python tests and 27 JavaScript tests. A separate float32 check on 890
tokens / 2,793 bytes of existing BabyLM validation text found maximum batched
versus independent difference 4.27e-8 bits/byte across all three untrained models.
No BabyLM test likelihoods or continuations were inspected for these checks.

The production preview shows the dated twelve-run snapshot, six selectable
validation components and only update counts shared by all three architectures.
Selecting Gutenberg changes the measured curve. Selecting the unstarted 100M
runs removes the curve and keeps comparisons unavailable. Exposure text correctly
reports approximately 99.1% and 9.7% of corpus token counts for the fixed budget,
explicitly distinguishing this ratio from unique-text coverage.

Desktop chart and table were visually inspected at a 1280-pixel viewport. At
390 pixels, document client/scroll widths both measured 375 pixels. The chart
reflows its viewBox to the 335.2-pixel drawing width, preserving readable labels;
tables stay within their horizontal scroll containers. All four controls remain
usable, and the browser still loads the completed WikiText checkpoint. Pending
BabyLM measurements are never rendered as zero scores or ranked against finished
runs. The new snapshot/protocol/prompt links resolve in the production build.

## Forward learning and physical-choice publication

The five-rule, three-seed study adds six Python checks for the normalized edge
derivative, exact gradients when omitted cross-neuron routes are absent, forward
state parity, fixed trace storage with episode length, reservoir isolation,
reward reproducibility and the shared task stream. The complete suite passes
52 Python tests. The 27 JavaScript tests pass after adding the new results view,
and the production build succeeds.

At 1280 pixels the two learning figures and two neural/physical figures load in
paired columns. Both five-row tables match their checked JSON records, including
the reward condition's failed reversal. There are no duplicate DOM IDs. At a
390-pixel viewport, client and document scroll widths both measure 375 pixels;
the learning figures stack at 335.2 pixels wide and the wider physical action
table scrolls inside its own container. The normal viewport was restored after
inspection. The language workspace still reports checkpoint 6,000 ready and
contains both 3D canvases.

The initial recorded MP4 stalled in this browser. Moving metadata to the start
with FFmpeg stream copy fixed playback; all 100 decoded frame checksums match
the original recording exactly. The published clip then played to completion
on the phone layout: duration 4 seconds, currentTime 4, ended true, readyState 4,
no media error. Original and published file hashes and the FFmpeg version are
recorded in the physical summary. It represents one simulated second at quarter
speed, with a learned high-level choice and a designed gait controller.

All four pages of the local-learning note were rendered and visually reviewed;
the final compiler log has no overflow or unresolved-reference warnings. The
source ZIP contains 43 files with a verified SHA-256 manifest. The BabyLM
validation snapshot now includes 9,500 measured updates in its first run; the
other eleven runs remain pending and no BabyLM test result is published.

Release `18ebaa3` passed GitHub Actions run `34429647837` and deployed to
`https://flm.kuber.studio`. The live browser shows both five-row tables and the
9,500-update BabyLM snapshot. The live physical clip also played to its exact
four-second end with readyState 4 and no media error. SHA-256 checks match local
bytes for both learning summaries, the video, four-page PDF, source archive,
declared learning protocol and BabyLM snapshot. This records the deployed
snapshot, not completion of the still-running BabyLM training queue.

## Language topology priority and completed wiring diagnostics

On 10 September, the language-control implementation passed 67 Python tests
and 42 JavaScript tests. New coverage checks independently crossed graph/model
seeds, identical initial parameters, actual no-slow retraining, graph-buffer and
RNG/exposure drift on resume, test gating before incomplete runs, complete
article denominators, and paired-effect signs. The browser comparison tests
reject mismatched updates and training seeds. The production build succeeds.

Historical/current numerical replay matches exactly for both WikiText seeds:
initial parameters, ten AdamW updates, sampled text and validation scoring.
Three independently generated 1,024-node nulls pass every declared invariant.
The language queue is running; this entry does not report control test scores.
BabyLM is paused with its first FLM run complete at 12,000 and GRU saved at 6,500.

The completed 60-run cue/context report verifies all checkpoint/report hashes,
all probability panels, matched sensory streams and initializations, and all 15
original measured-cue parameter bridges. All four new scientific figures were
rendered and visually inspected. The source ZIP now has 50 source files plus a
verified SHA-256 manifest; the separate 60-run archive retains every panel.

At desktop width 1280, Research renders eight language-control rows, five
wiring summaries and fifteen individual seed rows, without duplicate DOM IDs.
Switching to cue/delay 48 gives BPTT 100% versus 66.67%; restoring context/delay
8 gives 63.02% versus 76.04%, matching the verified data. The adjacency, context
accuracy and expanded gradient figures load correctly. At mobile width 390,
document and scroll widths are both 375 pixels; the 550-pixel language table
scrolls within its 335-pixel container. No page-wide overflow occurs.

The BabyLM browser model loads its actual checkpoint 12,000 and both 3D views.
A new 80-token continuation produces 272 UTF-8 bytes, live next-token
probabilities and nonzero recurrent state. The output contains malformed
phrases and invented words, retained as generated; this is not evidence of a
reliable assistant. Switching to WikiText restores a WikiText conversation;
switching back restores the BabyLM conversation. The numerical/worker tests
also cover tokenizer parity, generation, controls, adaptation and checkpoint
isolation for all three browser packages. Temporary viewport overrides are
reset after responsive review.

Release `da31299` passed Pages run `34454987903`. The public site shows eight
language-control rows, the matching 500-update comparison, all five context
result rows, the retained three-model WikiText test table and the paused BabyLM
snapshot. The public BabyLM checkpoint loads to ready with both 3D canvases.
Nine deployed protocol, result, figure, archive and model artifacts match their
local SHA-256 hashes byte for byte. The 71-entry wiring archive passes its CRC
check. The first language null run has also saved update 1,000 locally; that
later training progress is not retroactively part of the published 500-update
snapshot. No new control test loss has been scored.

## Wiring paper and completed-result gate

The five-page wiring-controls note is built from the reverified sixty-run
summary. Its generated table, three figures and source inputs have recorded
hashes. All five final page renders were inspected; the final LaTeX log has no
overflow or unresolved references. The reviewed PDF SHA-256 is
`7b1069907c3278573db44feceecf2a2292f6db41ea33245edb034308af5df177`.
The source archive now contains 57 source files plus its verified manifest.

The full JavaScript suite passes 44 tests. New checks reject missing or duplicate
conditions, incomplete article coverage, invalid checkpoints, wrong contrast
signs, inconsistent means and a result from a different study identity. The
three targeted Python scoring tests pass, including rejection before test-cache
decoding when any control is incomplete. The production build succeeds.

The preview shows the actual matched 2,000-update validation pair and keeps the
completed-test block hidden. Both links to the new wiring PDF are present, with
no duplicate DOM IDs. The completed-result tables await real test data for their
browser review; fixture checks are not presented as observed language outcomes.
An attached finalizer waits on the actual training process before invoking the
strict completion/selection gate and scoring pipeline. Training remains the
only active writer of control checkpoints.

## Exact language graph audit

The new structural view reproduces all four frozen 1,024-node/76,130-edge
graphs. Every null passes the invariant checks again, and the audit reproduces
the recorded topology statistics without reading losses or language text. The
4,096-row node CSV includes outgoing strength as well as preserved counts;
the graph ZIP passes its CRC and every manifest SHA-256 check. All four signed
matrices were visually reviewed in the generated 3,200-pixel scientific figure.

The full JavaScript suite now passes 46 tests. New structural tests bind the
display to the current study identity and reject missing graphs, duplicate
hashes, changed constraints, inconsistent components and asymmetric overlaps.
Both new Python tests pass, including a directed graph whose degrees are
preserved while outgoing weighted strengths change. Production build succeeds.

At the actual 598-pixel preview viewport, the expanded audit shows all four
correct graph rows, the loaded figure and three download links. Document and
scroll widths both equal 598; the table remains inside the research column.
The completed-test block stays hidden while language training is pending.
The attached preview surface did not honor viewport overrides. A separate
temporary test tab resolved that limitation: at desktop width 1280 the expanded
figure and all four rows render, with document and scroll widths both 1265. At
mobile width 390 they both equal 375, and the 550-pixel table scrolls inside its
335-pixel container. The figure loads and final-test results remain hidden.
The temporary tab was closed and the viewport override reset after review.

The previous paper release, commit `88ff8fd`, completed Pages run
`34456998206`. The deployed five-page PDF, source archive and 2,000-update
snapshot all matched local SHA-256 hashes. Both new paper links appeared in
the public Research view. This confirms publication, not language-study
completion.

The graph-audit release, `c643dc3`, completed Pages run `34458712555`. The live
Research view shows all four structural rows and all three download links.
Its saved update-4,000 pair reads measured 1.9340 versus rewired 1.9324 BPB,
difference +0.0016; that is partial validation, not a test conclusion. The
graph JSON, graph ZIP, node CSV, PNG/SVG figures and progress snapshot all match
local bytes; `reports/language-topology/structure-release.json` records their
verified SHA-256 hashes and deployment identity.

## Bounded training scheduler and preserved-state handoff

Six new scheduler tests pass, including a real competing process rejected by
the OS lease, release of that lease, the two-child bound, stopping new launches
after a failed child, verification-failure cleanup, memory-based serial fallback
and verified command/resume settings. Existing numerical sources are unchanged
and pass the frozen identity check. This addition does not change frontend code.

The legacy queue, child and finalizer were stopped after verifying the complete
5,000-update last/best checkpoint pair. The replacement queue owns two live
training processes with distinct registered outputs and four threads each.
Two replayed 100-update windows match the old loss, gradient norm, rate and
exposure fields exactly. `reports/language-topology/scheduler-transition.json`
records the checkpoint hashes, both old/new rows, new scheduler source hash,
actual process commands and resource observation. It does not claim an old
parameter-tensor comparison at unsaved intermediate updates. The dated public
snapshot is still gated on saved checkpoints, and no new control test was read.
