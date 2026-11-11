"""FastAPI HTTP control, PCM WebSocket input and WebRTC output."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, BeforeValidator, Field

from .config import FPS, RESOLUTIONS, ROOT, Settings
from .engine import DittoEngine
from .manager import Manager, SessionBusy
from .media import decode_audio
from .video_encoder import StableH264Encoder, install_encoder

Resolution = Annotated[Literal[512, 720, 1024, 1080, 1280, 1440, 1920, 2048, 2560, 3840], BeforeValidator(int)]
Gain = Annotated[float, Field(ge=0, le=1.5, allow_inf_nan=False)]


class Offer(BaseModel):
    type: Literal["offer"]
    sdp: str = Field(min_length=20, max_length=200000)


class Controls(BaseModel):
    motion_intensity: Gain | None = None
    mouth_intensity: Gain | None = None


async def read_upload(upload: UploadFile, limit: int):
    try:
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"Each upload is limited to {limit // 1024 // 1024} MiB.")
        if not data:
            raise HTTPException(422, "Upload is empty.")
        return data
    finally:
        await upload.close()


def create_app(settings=None, engine_factory=DittoEngine):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        settings.runtime.mkdir(parents=True, exist_ok=True)
        engine = await asyncio.to_thread(engine_factory, settings)
        app.state.manager = Manager(settings, engine)
        app.state.manager.reaper = asyncio.create_task(app.state.manager.cleanup())
        try:
            yield
        finally:
            await app.state.manager.close()

    app = FastAPI(
        title="Ditto Streaming API",
        version="1.0.0",
        lifespan=lifespan,
        description="Upload a portrait, select output resolution and motion intensity, then stream PCM speech to a persistent WebRTC avatar. One active session per worker/GPU. Models remain loaded across sessions.",
    )

    def manager():
        return app.state.manager

    def session(sid):
        s = manager().sessions.get(sid)
        if not s or s.closed:
            raise HTTPException(404, "Session not found or already closed.")
        return s

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/health")
    async def health():
        m = manager()
        return dict(
            ready=True,
            fps=FPS,
            face_decoder_size=512,
            resolutions=RESOLUTIONS,
            default_resolution=settings.default_resolution,
            video_mode=settings.video_mode,
            video_crf=settings.video_crf,
            motion_intensity_range=[0, 1.5],
            model_load_count=m.model_load_count,
            active_sessions=len(m.sessions),
            ice_servers=list(settings.ice_servers),
        )

    @app.post("/v1/sessions", status_code=201)
    async def create_session(
        avatar: Annotated[UploadFile, File(description="Portrait image. Proportions are preserved.")],
        resolution: Annotated[
            Resolution, Form(description="Output longest edge in pixels. Applied before fresh inference.")
        ] = 1024,
        motion_intensity: Annotated[
            float,
            Form(
                ge=0,
                le=1.5,
                allow_inf_nan=False,
                description="Overall motion gain: 0 freezes face motion, 1 is nominal, 1.5 exaggerates it. Also affects lips.",
            ),
        ] = 1.0,
        mouth_intensity: Annotated[float, Form(ge=0, le=1.5, allow_inf_nan=False)] = 0.65,
        audio: Annotated[
            UploadFile | None, File(description="Optional initial WAV/FLAC speech, up to 120 seconds.")
        ] = None,
    ):
        image = await read_upload(avatar, settings.max_upload_bytes)
        pcm = None
        if audio:
            data = await read_upload(audio, settings.max_upload_bytes)
            try:
                pcm = await asyncio.to_thread(decode_audio, data, settings.max_audio_seconds)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        try:
            s = await manager().create(image, resolution, motion_intensity, mouth_intensity)
        except SessionBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if pcm is not None:
            await s.append_audio(pcm)
        return s.status()

    @app.get("/v1/sessions/{sid}")
    async def get_status(sid: str):
        return session(sid).status()

    @app.delete("/v1/sessions/{sid}")
    async def delete_session(sid: str):
        s = session(sid)
        await s.close()
        return {"id": sid, "closed": True, "model_load_count": manager().model_load_count}

    @app.patch("/v1/sessions/{sid}/controls")
    async def controls(sid: str, body: Controls):
        s = session(sid)
        motion = s.motion_intensity if body.motion_intensity is None else body.motion_intensity
        mouth = s.mouth_intensity if body.mouth_intensity is None else body.mouth_intensity
        manager().engine.controls(motion, mouth)
        s.motion_intensity, s.mouth_intensity = motion, mouth
        return s.status()

    @app.post("/v1/sessions/{sid}/offer")
    async def offer(sid: str, body: Offer):
        s = session(sid)
        if s.pc:
            raise HTTPException(409, "An offer has already been accepted. Create another session to reconnect.")
        if "m=video " not in body.sdp or "m=audio " not in body.sdp:
            raise HTTPException(422, "Offer must receive both audio and video tracks.")
        pc = s.pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**v) for v in settings.ice_servers]))
        video_sender = pc.addTrack(s.video)
        s.video_encoder = StableH264Encoder(s.width, s.height, mode=settings.video_mode, crf=settings.video_crf)
        install_encoder(video_sender, s.video_encoder)
        pc.addTrack(s.audio)
        for tr in pc.getTransceivers():
            if tr.kind == "video":
                tr.setCodecPreferences(
                    [v for v in RTCRtpSender.getCapabilities("video").codecs if v.mimeType == "video/H264"]
                )

        @pc.on("connectionstatechange")
        async def changed():
            if pc.connectionState in ("closed", "failed") and not s.closed:
                asyncio.create_task(s.close())
            elif pc.connectionState == "disconnected":

                async def expire():
                    await asyncio.sleep(10)
                    if pc.connectionState == "disconnected":
                        await s.close()

                asyncio.create_task(expire())

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
            await pc.setLocalDescription(await pc.createAnswer())
        except Exception as exc:
            await s.close()
            raise HTTPException(422, "Could not negotiate this WebRTC offer.") from exc
        s.feed_task = asyncio.create_task(s.feed())
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.post("/v1/sessions/{sid}/audio", status_code=202)
    async def upload_audio(sid: str, audio: Annotated[UploadFile, File()]):
        s = session(sid)
        data = await read_upload(audio, settings.max_upload_bytes)
        try:
            pcm = await asyncio.to_thread(decode_audio, data, settings.max_audio_seconds)
            await s.append_audio(pcm)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"accepted_seconds": len(pcm) / 16000, "queued_audio_seconds": len(s.input_pcm) / 16000}

    @app.websocket("/v1/sessions/{sid}/audio")
    async def input_audio(ws: WebSocket, sid: str):
        s = manager().sessions.get(sid)
        if not s or s.closed:
            await ws.close(code=1008, reason="Session not found.")
            return
        if s.audio_socket:
            await ws.close(code=1008, reason="Another audio socket is active.")
            return
        s.audio_socket = True
        s.audio_ws = ws
        await ws.accept()
        try:
            while not s.closed:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    if not data or len(data) % 2 or len(data) > 128000:
                        raise ValueError("Send even-sized PCM16 packets, at most 128000 bytes each.")
                    await s.append_audio(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768)
                else:
                    control = json.loads(message.get("text", "{}"))
                    if control.get("type") != "end":
                        raise ValueError('Supported control: {"type":"end"}.')
                    s.input_active = False
                    await ws.send_json({"type": "ack", "stream_continues": True})
        except WebSocketDisconnect:
            pass
        except (ValueError, json.JSONDecodeError) as exc:
            await ws.close(code=1008, reason=str(exc)[:120])
        finally:
            s.audio_socket = False
            if s.audio_ws is ws:
                s.audio_ws = None
            s.input_active = False
        # Socket end is not session end. The avatar keeps rendering listening motion.

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


app = create_app()
