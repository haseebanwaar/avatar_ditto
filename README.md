# Ditto Streaming API

A standalone **FastAPI + WebRTC** service for a portrait-driven talking avatar. Upload a portrait, choose output resolution and motion intensity, then supply speech continuously. New silent-motion frames keep the avatar alive between utterances. GPU models stay resident across sessions.

This repository includes its Ditto inference source under `vendor/ditto`; it does not depend on a sibling project or on the original comparison dashboard. Model weights, uploaded portraits, generated media and virtual environments are excluded from Git.

## Features

- Avatar image upload with orientation normalization and preserved proportions.
- Output longest edge: **512, 720, 1024, 1080, 1280, 1440, 1920, 2048, 2560 or 3840 pixels**.
- **25 FPS** model clock and synchronized H.264 video / Opus audio over WebRTC.
- Stable H.264 image quality from startup, without encoder recreation on bitrate feedback.
- `motion_intensity` from **0 to 1.5**, adjustable during a session. `0` freezes facial movement, `1` is nominal, and larger values exaggerate movement. This scales head and facial motion, **including lips**.
- Independent `mouth_intensity`, default `0.65`, also adjustable live.
- Speech via WAV/FLAC uploads or **16 kHz, mono, signed PCM16 little-endian** WebSocket packets.
- Multiple utterances on the **same session and connection**. An audio `end` message does not stop the avatar.
- Browser demo at `/`, generated API documentation at `/docs`, OpenAPI at `/openapi.json`.
- One active session per process/GPU, with conflict responses, bounded queues and automatic cleanup of abandoned sessions.

**Resolution means composite canvas size.** The native generated face remains 512 x 512. Choosing 2K or 4K does not make the face decoder native 2K/4K. A half-body image is retained, but Ditto does not generate arm/torso gestures. This is an audio-driven avatar, not an LLM or a TTS service.

## Requirements

- Linux x86_64, Python **3.11**, an NVIDIA GPU and a driver compatible with CUDA 12.1.
- TensorRT **8.6.1**. The supplied engine bundle targets **Ampere Plus**, including the tested RTX 4090. Other devices or TensorRT versions can need separately built engines.
- A C/C++ compiler and Python development headers for the upstream Cython blend extension. On Ubuntu: `sudo apt-get install build-essential python3.11-dev python3.11-venv`.
- GPU and RAM capacity must be measured for the desired resolution and concurrency. This service uses one worker; additional Uvicorn workers each allocate their own models and do not share the session registry.

## Install with pip

```bash
bash scripts/bootstrap.sh
.venv/bin/python scripts/download_models.py
cp .env.example .env
.venv/bin/python scripts/serve.py
```

Open **http://127.0.0.1:7880/**. Upload a portrait and optionally a speech file, then connect. You can send more files or start/finish microphone input without reconnecting. Stop releases the session but keeps the models loaded.

`bootstrap.sh` creates `.venv`, installs the CUDA 12.1 PyTorch wheel, API/model dependencies and TensorRT, and extracts a separate cuDNN 8 runtime. This keeps TensorRT's cuDNN 8 requirement alongside PyTorch's cuDNN 9 runtime. The launcher sets the library search path before importing either runtime.

The downloader retrieves the official `digital-avatar/ditto-talkinghead` model bundle and checks every downloaded file against `model-manifest.json`. It downloads only the online config and Ampere Plus engines used here. The config is a pickle: load only trusted, verified checkpoint files. If upstream files change, checksum failure is intentional; review the change before updating the manifest. `--revision` accepts a Hugging Face revision for reproducible retrieval.

For existing trusted weights, set `DITTO_CHECKPOINTS=/absolute/path/to/checkpoints` instead of downloading again. This directory must contain `ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` and `ditto_trt_Ampere_Plus/`. No weights are copied into Git.

## API workflow

### 1. Create a session

```bash
curl -X POST http://127.0.0.1:7880/v1/sessions \
  -F 'avatar=@portrait.png' \
  -F 'resolution=2048' \
  -F 'motion_intensity=0.8' \
  -F 'mouth_intensity=0.65' \
  -F 'audio=@speech.wav'
```

`avatar` is required. `audio` is optional; omit it for a live or initially silent stream. HTTP **201** returns `id`, actual `width`/`height`, FPS, control values, `offer_url`, `audio_websocket`, `audio_upload_url` and `status_url`. The `resolution` form value defaults to **1024** if omitted. `DITTO_DEFAULT_RESOLUTION` sets the browser's initial selection only. Sizes preserve portrait aspect ratio and have even dimensions; the selected resolution is the longest edge, not a forced 16:9 frame.

No raw video is returned by this upload response. **Video/audio use WebRTC**, established in the next step. The browser demo implements the complete client.

### 2. Negotiate WebRTC

Create an `RTCPeerConnection` with **both audio and video receive transceivers**. Gather ICE candidates and POST its offer:

```http
POST /v1/sessions/SESSION_ID/offer
Content-Type: application/json

{"type":"offer","sdp":"...complete local SDP..."}
```

Set the returned `{type, sdp}` as the remote description. Receive synchronized audio/video tracks. One offer per session; create a new session to reconnect after failure. H.264 is selected for video. Configure `DITTO_ICE_SERVERS` for remote networks; the browser obtains the same ICE configuration from `/health`.

### 3. Supply speech repeatedly

Append another speech file without replacing the session:

```bash
curl -X POST http://127.0.0.1:7880/v1/sessions/SESSION_ID/audio \
  -F 'audio=@next-utterance.wav'
```

HTTP **202** means speech was queued. The queue has at most 120 seconds of pending audio; wait for it to drain before submitting more. Files are limited to 120 seconds and 32 MiB per upload. Speech audio is decoded, mixed to mono and resampled to 16 kHz. Prefer WAV or FLAC.

For real-time audio, open:

```text
ws://127.0.0.1:7880/v1/sessions/SESSION_ID/audio
```

Send binary PCM16 packets, preferably **100 ms / 3200 bytes** each, paced at the speaking rate. Each packet must have an even byte count and be at most 128000 bytes. Example:

```python
# ws is an established WebSocket; pcm16 is 16kHz mono signed little-endian audio.
await ws.send(pcm16.tobytes())
await ws.send('{"type":"end"}')
# Server acknowledges {"type":"ack","stream_continues":true}.
# Send the next utterance on the same socket whenever ready.
```

Missing input becomes generated silence. Sending `end`, closing the audio socket or finishing a file **does not close WebRTC**. The avatar continues rendering listening motion. Only one input WebSocket is accepted per session. HTTP uploads append to the same FIFO; avoid mixing a microphone and file upload simultaneously unless this is intended.

### 4. Adjust motion while streaming

```bash
curl -X PATCH http://127.0.0.1:7880/v1/sessions/SESSION_ID/controls \
  -H 'Content-Type: application/json' \
  -d '{"motion_intensity":0.6,"mouth_intensity":0.75}'
```

Updates are eased over successive frames to avoid abrupt pose jumps. Low overall gain also reduces lip articulation. Change the portrait or resolution by stopping and creating another session.

### 5. Inspect or close

```bash
curl http://127.0.0.1:7880/v1/sessions/SESSION_ID
curl -X DELETE http://127.0.0.1:7880/v1/sessions/SESSION_ID
```

Status includes dimensions, generated/played seconds, queued audio, frame buffer, stalls, error and model-load count. HTTP **409** indicates an existing active session. Unsupported parameters return **422**, excessive upload bytes **413**, and a missing/closed session **404**.

Session work files are deleted on close. There is no video recorder or public media directory in this service. A session without an offer expires after 60 seconds. Connected sessions expire after one hour by default; edit `.env` to change that limit. Video connection failure/disconnect also triggers cleanup. This supports continuous sessions within configured limits, not unlimited storage or unlimited concurrency.

## Video quality and network rate

The default `DITTO_VIDEO_MODE=quality` uses constant-quality H.264 (CRF 20), a 25 FPS clock, one-second keyframes, a faster encoder preset and a single encoder context throughout the session. Playback waits for eight generated frames, and inference can run up to 1.2 seconds ahead to absorb short scheduling/encoding delays. It avoids aiortc's default 1 Mbps startup / 3 Mbps ceiling and bitrate-feedback-driven encoder resets, which produced cyclic blocking artifacts in a 2K reproduction. No global/site-packages codec patch is installed.

This default is intended for localhost or a sufficiently fast LAN. It deliberately does **not** follow receiver REMB rate estimates: actual bandwidth varies with image detail and motion. Use `DITTO_VIDEO_MODE=adaptive` for a constrained remote network. Adaptive mode respects feedback with a resolution-aware ceiling and updates rate control without destroying reference frames; insufficient bandwidth can still reduce quality. It cannot promise both full-resolution quality and real-time delivery on a slow link.

`DITTO_VIDEO_CRF` defaults to `20` (allowed 16-30; lower means higher quality and more bandwidth). Resolution and 25 FPS are preserved. Session status includes `video_encoder` counters (mode, requested/target bitrate, context creations, encoded frames/bytes), so repeated resets can be distinguished from inference or network issues.

The per-sender encoder hook is guarded against **aiortc 1.16.0**. Keep that pin, or review and test the hook when upgrading. It uses the inherited RTP packetization; only encoding/rate behavior changes.

To compare the stock and stable encoders against the same source video:

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_encoders.py --video reference.mp4
```

For a three-minute GPU/WebRTC startup test, add `--duration 180` to `scripts/verify_stream.py`.

## Configuration

Copy `.env.example`. Key settings:

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU visible to the worker |
| `DITTO_HOST` / `DITTO_PORT` | `127.0.0.1` / `7880` | HTTP bind |
| `DITTO_CHECKPOINTS` | `checkpoints` | Verified model directory |
| `DITTO_RUNTIME` | `runtime` | Temporary session files |
| `DITTO_DEFAULT_RESOLUTION` | `1024` | Initial browser selection |
| `DITTO_CONNECT_TIMEOUT` | `60` | Seconds allowed to negotiate an offer |
| `DITTO_MAX_SESSION_SECONDS` | `3600` | Session lifetime cap |
| `DITTO_ICE_SERVERS` | `[]` | JSON array of browser/aiortc ICE server settings |
| `DITTO_CUDNN8_DIR` | `.runtime/cudnn8/nvidia/cudnn/lib` | Optional CUDA compatibility library override |

Bind is local by default. Public deployment requires your own access control, HTTPS/WSS, upload/request limits at the proxy, a suitable STUN/TURN setup and routing each client's requests to the same single-worker instance. Browser microphone access requires HTTPS or localhost. Do not expose this unauthenticated development service directly to the internet. No public tunnel is created by the launcher.

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/ruff check ditto_api scripts tests
```

API tests use an injected fake engine; no GPU or downloaded weights are required, but runtime Python dependencies must be installed. They cover upload/parameter validation, aspect ratios, conflicts, cleanup, model residency, repeated speech and WebSocket end semantics. A keypoint test verifies overall motion gain at zero, nominal and amplified settings.

With the GPU server running, use your own portrait and at least four seconds of 16kHz WAV speech:

```bash
.venv/bin/python scripts/verify_stream.py \
  --avatar portrait.png --audio speech-16k.wav --resolution 2048
```

This independently decodes WebRTC while sending paced PCM, pauses, changes motion intensity, uploads another utterance and verifies moving idle frames, dimensions, at least 20 received FPS, zero generation stalls and one model load. Results go to ignored `runtime/verification.json`. This is a short, local, single-stream test, not a production load test or a lip-sync accuracy score.

## Source and licensing

- API and demo source are distributed under Apache 2.0; see `LICENSE` and `NOTICE`.
- Vendored Ditto inference: [Ant Group / ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead), commit `c3e47eee2e626500017a0556b470d6d4182f85e8`. Its original license is retained under `vendor/ditto/LICENSE`.
- Model source: [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
- Overall motion intensity is implemented by scaling final stitched keypoint displacement about the source pose; mouth strength is a separate expression gain. The modified upstream file is identified in `NOTICE`.
- **Commercial clearance is not implied.** Exact weights and third-party components require separate review. [InsightFace](https://github.com/deepinsight/insightface#license) distinguishes its MIT code from pretrained models with non-commercial research restrictions. License or replace restricted weights before commercial deployment.

## Upload to Git

The directory is its own Git repository with no remote configured. Review the source and commit it, then add your remote and push. Do not add checkpoints, `.env`, `.venv`, `runtime`, CUDA runtimes or private input media. `.gitignore` excludes these. A source-only ZIP can also be uploaded or extracted into a new Git checkout.
