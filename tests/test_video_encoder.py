from fractions import Fraction

import av
import numpy as np
from aiortc import RTCPeerConnection

from ditto_api.video_encoder import StableH264Encoder, bitrate_for_size, install_encoder


def make_frame(index):
    image = np.random.default_rng(5).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    image[30:50, index : index + 20] = 255
    f = av.VideoFrame.from_ndarray(image, format="rgb24")
    f.pts, f.time_base = index, Fraction(1, 25)
    return f


def test_quality_feedback_does_not_restart_encoder():
    encoder = StableH264Encoder(128, 128)
    identity = None
    for i, feedback in enumerate((500000, 900000, 3000000, 600000, 8000000)):
        encoder.target_bitrate = feedback
        payloads, _ = encoder.encode(make_frame(i))
        assert payloads
        if identity is None:
            identity = encoder.codec
        assert encoder.codec is identity
    assert encoder.context_creations == 1
    assert encoder.target_bitrate == encoder.budget
    assert encoder.requested_bitrate == 8000000


def test_adaptive_rate_changes_retain_references():
    encoder = StableH264Encoder(128, 128, mode="adaptive")
    encoder.encode(make_frame(0))
    context = encoder.codec
    encoder.target_bitrate = 400000
    encoder.encode(make_frame(1))
    assert encoder.codec is context
    assert encoder.codec.bit_rate == 400000
    assert encoder.context_creations == 1


def test_resolution_budget_scales():
    assert bitrate_for_size(1536, 2048) > 3000000
    assert bitrate_for_size(3840, 2160) > bitrate_for_size(1536, 2048)


def test_encoder_install_is_per_sender():
    pc = RTCPeerConnection()
    sender = pc.addTransceiver("video", direction="sendonly").sender
    encoder = StableH264Encoder(128, 128)
    install_encoder(sender, encoder)
    assert sender._RTCRtpSender__encoder is encoder
