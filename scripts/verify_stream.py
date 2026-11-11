"""GPU integration test: paced speech, silence and another utterance on one stream."""

import argparse
import asyncio
import io
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


async def main(args):
    origin = time.monotonic()
    samples, idle, tasks = [], [], []
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    async with aiohttp.ClientSession() as http:
        form = aiohttp.FormData()
        form.add_field("avatar", Path(args.avatar).read_bytes(), filename="avatar.png", content_type="image/png")
        form.add_field("resolution", str(args.resolution))
        form.add_field("motion_intensity", ".8")
        async with http.post(args.url + "/v1/sessions", data=form) as r:
            assert r.status == 201, await r.text()
            state = await r.json()
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")

        async def receive(track):
            try:
                while True:
                    frame = await track.recv()
                    if track.kind == "video":
                        media = float(frame.pts * frame.time_base)
                        samples.append(
                            {
                                "wall": time.monotonic() - origin,
                                "media": media,
                                "width": frame.width,
                                "height": frame.height,
                            }
                        )
                        if 5 <= media <= 5.8:
                            idle.append(frame.reformat(width=96, height=96).to_ndarray(format="rgb24"))
            except MediaStreamError:
                pass

        @pc.on("track")
        def on_track(track):
            tasks.append(asyncio.create_task(receive(track)))

        try:
            await pc.setLocalDescription(await pc.createOffer())
            async with http.post(
                args.url + state["offer_url"], json={"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}
            ) as r:
                assert r.status == 200, await r.text()
                answer = await r.json()
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            assert sr == 16000 and len(audio) >= 4 * sr
            audio = audio[: 4 * sr]
            async with http.ws_connect(args.url.replace("http", "ws", 1) + state["audio_websocket"]) as ws:
                sent = time.monotonic()
                for off in range(0, len(audio), 1600):
                    await asyncio.sleep(max(0, sent + off / sr - time.monotonic()))
                    await ws.send_bytes(np.clip(audio[off : off + 1600] * 32767, -32768, 32767).astype("<i2").tobytes())
                await ws.send_json({"type": "end"})
                assert (await ws.receive_json())["stream_continues"]
                await asyncio.sleep(2)
                async with http.patch(
                    args.url + state["status_url"] + "/controls", json={"motion_intensity": 0.5}
                ) as r:
                    assert r.status == 200, await r.text()
                b = io.BytesIO()
                sf.write(b, audio, sr, format="WAV")
                form = aiohttp.FormData()
                form.add_field("audio", b.getvalue(), filename="next.wav", content_type="audio/wav")
                async with http.post(args.url + state["audio_upload_url"], data=form) as r:
                    assert r.status == 202, await r.text()
                await asyncio.sleep(7)
                # Optional multi-minute soak with repeated speech on one connection.
                deadline = sent + args.duration
                while time.monotonic() < deadline:
                    await asyncio.sleep(min(4, max(0, deadline - time.monotonic())))
                    if time.monotonic() >= deadline:
                        break
                    form = aiohttp.FormData()
                    form.add_field("audio", b.getvalue(), filename="repeat.wav", content_type="audio/wav")
                    async with http.post(args.url + state["audio_upload_url"], data=form) as r:
                        assert r.status == 202, await r.text()
            async with http.get(args.url + state["status_url"]) as r:
                final = await r.json()
            assert not final["error"] and final["stalls"] == 0, final
            assert final["model_load_count"] == 1 and final["motion_intensity"] == 0.5
            assert samples and max(samples[-1]["width"], samples[-1]["height"]) == args.resolution
            steady = [v for v in samples if 2 <= v["media"] <= 10]
            fps = (len(steady) - 1) / (steady[-1]["wall"] - steady[0]["wall"])
            assert fps >= 20, fps
            movement = float(np.mean(np.abs(idle[-1].astype(float) - idle[0].astype(float))))
            assert movement > 0, "Idle animation froze"
            result = dict(
                resolution=args.resolution,
                dimensions=[final["width"], final["height"]],
                received_fps=fps,
                first_decoded_video_seconds_including_upload_setup=samples[0]["wall"],
                video_frames=len(samples),
                idle_pixel_difference=movement,
                stalls=final["stalls"],
                model_load_count=final["model_load_count"],
                repeated_speech_same_session=True,
                dynamic_motion_intensity=final["motion_intensity"],
                frame_arrivals=samples,
                encoder=final.get("video_encoder"),
                soak_duration_seconds=args.duration,
            )
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k != "frame_arrivals"}, indent=2))
        finally:
            async with http.delete(args.url + state["status_url"]) as r:
                assert r.status in (200, 404), await r.text()
            await pc.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:7880")
    p.add_argument("--avatar", required=True)
    p.add_argument("--audio", required=True, help="At least 4 seconds of WAV speech at 16kHz")
    p.add_argument("--resolution", type=int, default=2048)
    p.add_argument(
        "--duration", type=int, default=13, help="Total paced test duration; use 180 for startup quality soak"
    )
    p.add_argument("--output", default="runtime/verification.json")
    asyncio.run(main(p.parse_args()))
