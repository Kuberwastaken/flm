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
