# Browser validation log

Local desktop browser, checkpoint 3,000, 10 September 2026. These are functional checks, not held-out release benchmarks.

- Browser generation produced 400 bytes from the actual worker, with changing neural state and next-byte probabilities. An earlier 1,000-step checkpoint produced approximately 105 bytes/s including the deliberately slowed observation playback; this is a single interactive observation, not a standardized throughput benchmark.
- Anatomy and checkpoint checksums verified before use. Both WebGL views rendered without captured console errors. The brain framing was corrected to exclude the VNC from the default anatomical context crop; all 1,024 model neurons remain in inference. Five selected neurons have no soma coordinate and remain inspectable through the selector.
- The articulated body loaded all 39 mesh files. Automated FK tests match all reference joint coordinates. Body pose is explicitly illustrative.
- Local adaptation test: two passes over 139 UTF-8 bytes of original test dialogue, rate 0.03; a separate 50-byte probe improved from 3.737 to 2.674 bits/byte. Online training loss was 2.718 bits/byte. This small, related probe is not evidence of broad generalization. The UI reported 278 completed updates and successfully saved the adapter in browser storage.
- At the 390-pixel responsive viewport setting, the browser's document client width and scroll width were both 375 pixels: no horizontal document overflow. Desktop layout places conversation on the left and brain/body on the right; narrow layout stacks them.

Outstanding release checks include production-bundle behavior, cancellation, intervention replay, storage roundtrip, keyboard navigation, failed asset handling, download links and the deployed custom domain. Add verified results here as those checks complete.
