"""Compare encoding quality against source frames while bitrate feedback changes."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from aiortc.codecs.h264 import H264Encoder

from ditto_api.video_encoder import StableH264Encoder

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--frames", type=int, default=150)
p.add_argument("--output", default="runtime/encoder-comparison.json")
a = p.parse_args()
source = av.open(a.video)
width = source.streams.video[0].width
height = source.streams.video[0].height
encoders = {"upstream": H264Encoder(), "stable": StableH264Encoder(width, height)}
decoders = {k: av.CodecContext.create("h264", "r") for k in encoders}
rows = {k: [] for k in encoders}
contexts = {k: [] for k in encoders}
feedback = (1000000, 500000, 1600000, 700000, 3000000, 600000)
for i, frame in enumerate(source.decode(video=0)):
    if i >= a.frames:
        break
    reference = frame.to_ndarray(format="rgb24")
    reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    for name, encoder in encoders.items():
        encoder.target_bitrate = feedback[(i // 25) % len(feedback)]
        frame.pts, frame.time_base = i, Fraction(1, 25)
        nals = list(encoder._encode_frame(frame, False))
        if not contexts[name] or contexts[name][-1] is not encoder.codec:
            contexts[name].append(encoder.codec)
        decoded = decoders[name].decode(av.Packet(b"".join(b"\x00\x00\x00\x01" + n for n in nals)))
        assert len(decoded) == 1
        actual = cv2.cvtColor(decoded[0].to_ndarray(format="rgb24"), cv2.COLOR_RGB2GRAY)
        y0, y1, x0, x1 = int(height * 0.15), int(height * 0.7), int(width * 0.2), int(width * 0.8)
        quality = cv2.PSNR(reference[y0:y1, x0:x1], actual[y0:y1, x0:x1])
        rows[name].append(quality)
        if i in (1, 26, 76):
            dest = Path(a.output).parent
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(dest / f"{name}-{i}.jpg"), cv2.cvtColor(decoded[0].to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
            )
result = {
    k: dict(
        context_creations=len(contexts[k]),
        mean_face_luma_psnr=float(np.mean(v)),
        min_face_luma_psnr=float(min(v)),
        psnr_per_frame=v,
    )
    for k, v in rows.items()
}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps({k: {f: v for f, v in d.items() if f != "psnr_per_frame"} for k, d in result.items()}, indent=2))
