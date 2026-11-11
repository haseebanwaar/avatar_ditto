import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (512, 720, 1024, 1080, 1280, 1440, 1920, 2048, 2560, 3840)
FPS = 25
SR = 16000


@dataclass(frozen=True)
class Settings:
    checkpoints: Path
    runtime: Path
    default_resolution: int = 1024
    max_session_seconds: int = 3600
    connect_timeout: int = 60
    max_upload_bytes: int = 32 * 1024 * 1024
    max_audio_seconds: int = 120
    ice_servers: tuple = ()
    video_mode: str = "quality"
    video_crf: int = 20

    @classmethod
    def from_env(cls):
        def path(name, fallback):
            p = Path(os.getenv(name, fallback)).expanduser()
            return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()

        resolution = int(os.getenv("DITTO_DEFAULT_RESOLUTION", "1024"))
        if resolution not in RESOLUTIONS:
            raise ValueError("Unsupported DITTO_DEFAULT_RESOLUTION")
        mode = os.getenv("DITTO_VIDEO_MODE", "quality")
        crf = int(os.getenv("DITTO_VIDEO_CRF", "20"))
        if mode not in ("quality", "adaptive") or not 16 <= crf <= 30:
            raise ValueError("Use DITTO_VIDEO_MODE=quality|adaptive and DITTO_VIDEO_CRF=16..30")
        return cls(
            checkpoints=path("DITTO_CHECKPOINTS", "checkpoints"),
            runtime=path("DITTO_RUNTIME", "runtime"),
            default_resolution=resolution,
            max_session_seconds=int(os.getenv("DITTO_MAX_SESSION_SECONDS", "864000")),
            connect_timeout=int(os.getenv("DITTO_CONNECT_TIMEOUT", "60")),
            video_mode=mode,
            video_crf=crf,
            ice_servers=tuple(json.loads(os.getenv("DITTO_ICE_SERVERS", "[]"))),
        )
