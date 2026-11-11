"""Validate uploads and preserve portrait aspect ratios."""

import io
import math
import warnings

import numpy as np
import soundfile as sf
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.signal import resample_poly

from .config import SR


def prepare_portrait(data: bytes, resolution: int, destination):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError("Portrait must contain at most 40 million pixels.")
                image = ImageOps.exif_transpose(source).convert("RGB")
                scale = resolution / max(image.size)
                width = max(2, round(image.width * scale / 2) * 2)
                height = max(2, round(image.height * scale / 2) * 2)
                image.resize((width, height), Image.Resampling.LANCZOS).save(destination)
                return width, height
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Upload a valid PNG, JPEG or other Pillow-supported portrait.") from exc


def decode_audio(data: bytes, max_seconds=120):
    try:
        with sf.SoundFile(io.BytesIO(data)) as source:
            if source.samplerate < 8000 or source.samplerate > 192000:
                raise ValueError("Unsupported audio sample rate.")
            if source.frames / source.samplerate > max_seconds:
                raise ValueError(f"Audio uploads must be at most {max_seconds} seconds.")
            audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
            sr = source.samplerate
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("Upload a WAV or FLAC speech file.") from exc
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite samples.")
    if sr != SR:
        factor = math.gcd(sr, SR)
        audio = resample_poly(audio, SR // factor, sr // factor)
    return np.clip(audio, -1, 1).astype(np.float32)
