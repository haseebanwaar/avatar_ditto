"""Bounded continuous PCM -> Ditto -> synchronized WebRTC tracks."""

import asyncio
import contextlib
import time
from collections import deque
from fractions import Fraction

import av
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from .config import FPS, SR


class Session:
    def __init__(self, manager, folder, resolution, width, height, motion, mouth):
        self.manager, self.folder = manager, folder
        self.id = folder.name
        self.resolution, self.width, self.height = resolution, width, height
        self.motion_intensity, self.mouth_intensity = motion, mouth
        self.created = time.monotonic()
        self.loop = asyncio.get_running_loop()
        self.condition = asyncio.Condition()
        self.input_pcm = np.zeros(0, np.float32)
        self.past = np.zeros(1920, np.float32)
        self.timeline = np.zeros(0, np.float32)
        self.timeline_base = 0
        self.frames = deque()
        self.ready_frames = self.video_index = self.audio_index = self.fed_samples = 0
        self.origin = self.first_ready = None
        self.stalls = 0
        self.stall_started = None
        self.stall_seconds = 0.0
        self.error = None
        self.closed = False
        self.pc = self.feed_task = None
        self.video_encoder = None
        self.audio_socket = False
        self.audio_ws = None
        self.input_active = False
        self.video, self.audio = Video(self), Audio(self)

    def status(self):
        return dict(
            id=self.id,
            resolution=self.resolution,
            width=self.width,
            height=self.height,
            fps=FPS,
            face_decoder_size=512,
            motion_intensity=self.motion_intensity,
            mouth_intensity=self.mouth_intensity,
            closed=self.closed,
            error=self.error,
            model_load_count=self.manager.model_load_count,
            video_encoder=self.video_encoder.status() if self.video_encoder else None,
            played_seconds=self.video_index / FPS,
            generated_seconds=self.ready_frames / FPS,
            buffer_seconds=max(0, (self.ready_frames - self.video_index) / FPS),
            queued_audio_seconds=len(self.input_pcm) / SR,
            input_active=self.input_active,
            first_frame_seconds=self.first_ready - self.created if self.first_ready else None,
            stalls=self.stalls,
            stall_seconds=self.stall_seconds,
            offer_url=f"/v1/sessions/{self.id}/offer",
            audio_websocket=f"/v1/sessions/{self.id}/audio",
            audio_upload_url=f"/v1/sessions/{self.id}/audio",
            status_url=f"/v1/sessions/{self.id}",
        )

    async def notify(self):
        async with self.condition:
            self.condition.notify_all()

    async def append_audio(self, pcm):
        if self.closed:
            raise ValueError("Session is closed.")
        async with self.condition:
            # Bounded even for a long-lived live session. Back-to-back uploads must drain.
            if len(self.input_pcm) + len(pcm) > self.manager.settings.max_audio_seconds * SR:
                raise ValueError("Audio queue is full; wait for queued speech to play.")
            self.input_pcm = np.concatenate((self.input_pcm, pcm))
            self.input_active = True
            self.condition.notify_all()

    def on_frame(self, rgb):
        if not self.closed:
            self.loop.call_soon_threadsafe(self.publish, rgb.copy())

    def publish(self, rgb):
        if self.closed:
            return
        if len(self.frames) >= 50:
            self.error = "Playback is too slow; reconnect the stream."
            asyncio.create_task(self.notify())
            return
        if self.first_ready is None:
            self.first_ready = time.monotonic()
        self.frames.append(rgb)
        self.ready_frames += 1
        if self.stall_started is not None:
            delay = max(0, time.monotonic() - self.stall_started)
            self.origin += delay
            self.stall_seconds += delay
            self.stall_started = None
        asyncio.create_task(self.notify())

    async def feed(self):
        try:
            while not self.closed:
                async with self.condition:
                    # Limit inference lead and prevent unbounded video/audio queues.
                    await self.condition.wait_for(
                        lambda: self.closed or self.error or self.fed_samples / SR < self.video_index / FPS + 1.2
                    )
                    if self.closed or self.error:
                        break
                    # A small lookahead buffer. Empty input becomes fresh silent motion.
                    if len(self.input_pcm) < 4560:
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(
                                self.condition.wait_for(lambda: self.closed or len(self.input_pcm) >= 4560), 0.1
                            )
                    if self.closed:
                        break
                    current = np.pad(self.input_pcm[:3200], (0, max(0, 3200 - len(self.input_pcm))))
                    preview = self.input_pcm[3200:4560]
                    future = np.pad(preview, (0, 1360 - len(preview)))
                    chunk = np.concatenate((self.past, current, future)).astype(np.float32)
                    self.past = current[-1920:].copy()
                    self.input_pcm = self.input_pcm[min(3200, len(self.input_pcm)) :]
                    self.timeline = np.concatenate((self.timeline, current))
                    self.fed_samples += 3200
                await asyncio.to_thread(self.manager.engine.run, chunk)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.error = str(exc)
            await self.notify()

    async def wait(self, end_frame, media_time):
        async with self.condition:
            required = max(end_frame, 8) if self.origin is None else end_frame
            while self.ready_frames < required and not self.closed and not self.error:
                if self.origin is not None:
                    due = self.origin + media_time
                    remaining = due - time.monotonic()
                    if remaining > 0:
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(self.condition.wait(), remaining)
                        continue
                    if self.stall_started is None:
                        self.stalls += 1
                        self.stall_started = due
                await self.condition.wait()
            if self.closed or self.error:
                raise MediaStreamError
            if self.origin is None:
                self.origin = time.monotonic() + 0.12

    async def pace(self, t):
        await asyncio.sleep(max(0, self.origin + t - time.monotonic()))

    async def close(self):
        if self.closed:
            return
        self.closed = True
        await self.notify()
        if self.feed_task:
            self.feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.feed_task
        try:
            await asyncio.to_thread(self.manager.engine.close_session)
        finally:
            if self.audio_ws:
                with contextlib.suppress(RuntimeError):
                    await self.audio_ws.close(code=1000, reason="Session closed.")
            self.video.stop()
            self.audio.stop()
            if self.pc:
                await self.pc.close()
            self.frames.clear()
            self.input_pcm = self.timeline = np.zeros(0, np.float32)
            self.manager.release(self)


class Video(MediaStreamTrack):
    kind = "video"

    def __init__(self, session):
        super().__init__()
        self.s = session

    async def recv(self):
        s = self.s
        await s.wait(s.video_index + 1, s.video_index / FPS)
        await s.pace(s.video_index / FPS)
        rgb = s.frames.popleft()
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = s.video_index * 3600, Fraction(1, 90000)
        s.video_index += 1
        await s.notify()
        return frame


class Audio(MediaStreamTrack):
    kind = "audio"

    def __init__(self, session):
        super().__init__()
        self.s = session

    async def recv(self):
        s = self.s
        await s.wait((s.audio_index + 960 + 1919) // 1920, s.audio_index / 48000)
        await s.pace(s.audio_index / 48000)
        position = s.audio_index // 3 - s.timeline_base
        samples = s.timeline[position : position + 321]
        pcm = (
            np.interp(np.arange(960) / 3, np.arange(len(samples)), samples, left=0, right=0)
            if len(samples)
            else np.zeros(960)
        )
        frame = av.AudioFrame.from_ndarray(
            np.clip(pcm * 32767, -32768, 32767).astype(np.int16)[None], format="s16", layout="mono"
        )
        frame.pts, frame.time_base, frame.sample_rate = s.audio_index, Fraction(1, 48000), 48000
        s.audio_index += 960
        # Keep one second of history, not the entire session's audio.
        keep = max(s.timeline_base, min(s.audio_index // 3, s.video_index * 640) - SR)
        if keep > s.timeline_base:
            s.timeline = s.timeline[keep - s.timeline_base :]
            s.timeline_base = keep
        return frame
