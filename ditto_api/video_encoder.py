"""Per-session H.264 encoder with stable quality and no bitrate-triggered resets.

aiortc 1.16.0's sender has no public encoder-factory hook. Installation uses
its private encoder slot, guarded by an exact version check. No site-packages
or global codec factories are modified.
"""

import fractions
import time

import av
from aiortc.codecs.h264 import H264Encoder


def bitrate_for_size(width, height):
    # Resolution-aware budget, at 25 FPS. Target is an ABR target only in adaptive mode.
    pixels = width * height
    return max(1_500_000, min(24_000_000, round(pixels * 2.6 / 100_000) * 100_000))


class StableH264Encoder(H264Encoder):
    def __init__(self, width, height, mode="quality", crf=20):
        super().__init__()
        self.mode = mode
        self.crf = crf
        self.budget = bitrate_for_size(width, height)
        self._bitrate = self.budget
        self.requested_bitrate = None
        self.context_creations = 0
        self.encoded_frames = 0
        self.encoded_bytes = 0
        self.last_rate_update = 0.0

    @property
    def target_bitrate(self):
        return self._bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate):
        self.requested_bitrate = int(bitrate)
        # LAN quality mode deliberately avoids the receiver's startup REMB ramp.
        # Use adaptive mode on constrained links; it may trade image quality for rate.
        if self.mode == "adaptive":
            target = max(300_000, min(int(bitrate), self.budget))
            now = time.monotonic()
            if target < self._bitrate or now - self.last_rate_update >= 1:
                self._bitrate = target
                self.last_rate_update = now

    def _encode_frame(self, frame, force_keyframe):
        if self.codec and (frame.width != self.codec.width or frame.height != self.codec.height):
            raise ValueError("Video dimensions cannot change inside a session.")
        if self.codec is None:
            self.codec = av.CodecContext.create("libx264", "w")
            self.codec.width, self.codec.height = frame.width, frame.height
            self.codec.pix_fmt = "yuv420p"
            self.codec.framerate = fractions.Fraction(25, 1)
            self.codec.time_base = fractions.Fraction(1, 25)
            self.codec.bit_rate = self.target_bitrate
            self.codec.gop_size = 25
            self.codec.max_b_frames = 0
            self.codec.thread_count = 4
            self.codec.profile = "Baseline"
            self.codec.options = {"preset": "veryfast", "tune": "zerolatency"}
            if self.mode == "quality":
                self.codec.options.update(crf=str(self.crf), **{"x264-params": "keyint=25:min-keyint=25:scenecut=0"})
            # Let x264 choose a level that actually supports the frame size/rate.
            # The upstream fixed level 3.1 is insufficient for a 2K portrait.
            self.context_creations += 1
        if self.mode == "adaptive":
            # FFmpeg/libx264 reconfigures rate control in place; retain reference frames.
            self.codec.bit_rate = self.target_bitrate
        frame.pict_type = av.video.frame.PictureType.I if force_keyframe else av.video.frame.PictureType.NONE
        data = b"".join(bytes(packet) for packet in self.codec.encode(frame))
        self.encoded_frames += 1
        self.encoded_bytes += len(data)
        if data:
            yield from self._split_bitstream(data)

    def status(self):
        return dict(
            mode=self.mode,
            target_bitrate=self.target_bitrate,
            receiver_requested_bitrate=self.requested_bitrate,
            crf=self.crf if self.mode == "quality" else None,
            context_creations=self.context_creations,
            encoded_frames=self.encoded_frames,
            encoded_bytes=self.encoded_bytes,
        )


def install_encoder(sender, encoder):
    from importlib.metadata import version

    if version("aiortc") != "1.16.0":
        raise RuntimeError(
            "Custom encoder integration requires aiortc==1.16.0; review the sender hook before upgrading."
        )
    name = "_RTCRtpSender__encoder"
    if not hasattr(sender, name) or getattr(sender, name) is not None:
        raise RuntimeError("Unexpected aiortc encoder state.")
    setattr(sender, name, encoder)
