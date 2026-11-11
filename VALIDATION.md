# Local validation

Test date: 11 November 2026.

- Independent WebRTC receiver decoded a fresh 2K composite at approximately **24.98 FPS**.
- Output: **1536 x 2048**, 25 FPS model clock.
- **Zero generation stalls**, one model load.
- A paced PCM utterance, generated silence and a second uploaded utterance used the same session.
- Overall motion intensity changed from 0.8 to 0.5 without reconnecting.
- Idle-frame pixel differences confirmed ongoing animation during a speech pause.
- First decoded video arrived in approximately **1.26 seconds including portrait upload/setup**. This is not a live audio-only latency number.
- Browser smoke test also confirmed uploaded-portrait playback at 2K. One generation buffer pause was observed in that separate interactive run; the automated zero-stall result is not a guarantee for every browser session.
- Physical GPU 1 (RTX 4090), TensorRT 8.6.1; local loopback HTTP/WebSocket/WebRTC.

`tests/` covers API validation, conflicts, aspect ratios, upload errors, repeated audio, WebSocket lifecycle, session expiry, model residency and keypoint motion scaling. `scripts/verify_stream.py` reproduces the GPU streaming check with your input files.

The API ran in its own pip virtual environment, reusing locally installed GPU dependencies and verified checkpoint files for this execution. A completely fresh CUDA/TensorRT installation via `scripts/bootstrap.sh` was not repeated on this machine. Its steps follow the working pip-based installation. Hardware, driver, engine compatibility and third-party weight terms still need review on another machine.

## Startup video quality correction

The stock aiortc H.264 encoder started at 1 Mbps, capped its target at 3 Mbps and recreated the codec after target changes over 10%. A controlled 150-frame 2K comparison with scripted receiver feedback reproduced strong blocking artifacts. The stock encoder created six contexts; the service encoder created one. Mean face-region luma PSNR against the unmodified source frames improved from 33.83 dB to 44.94 dB; the minimum improved from 29.35 dB to 44.23 dB. These are compression-fidelity measurements, not a measure of Ditto lip-sync or native generation quality.

The default now uses a CRF-20 quality encoder, a 25 FPS clock, one-second keyframes and no feedback-triggered context recreation. An optional adaptive mode updates rate control in place. No installed aiortc source files were changed. An eight-frame startup buffer and 1.2-second inference lead address short buffer underruns identified during the longer test. Quality mode favors localhost/LAN image fidelity and may use more bandwidth; it does not adapt to receiver REMB estimates.

Final buffered 180-second run: 25.00 received FPS overall, 24.95-25.02 FPS across 10-second windows, zero generation stalls, one encoder context and one model load. First decoded video, including upload/setup: 1.39 seconds. See `encoder-validation.json`. The interactive 2048 x 1364 browser stream also confirmed the updated encoder and zero stalls during the final check.
