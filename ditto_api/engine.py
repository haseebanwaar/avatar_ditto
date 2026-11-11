"""Adapter around vendored upstream Ditto. Weights stay loaded across sessions."""

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np

from .config import ROOT


class Quiet:
    def update(self, *args):
        pass

    def close(self):
        pass


class CallbackWriter:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, frame, fmt="rgb"):
        self.callback(frame if fmt == "rgb" else frame[:, :, ::-1])

    def close(self):
        pass


class CachedRegistrar:
    def __init__(self, wrapped):
        self.wrapped, self.cache = wrapped, {}

    def __call__(self, path, **kwargs):
        key = hashlib.sha256(Path(path).read_bytes()).hexdigest() + repr(sorted(kwargs.items()))
        if key not in self.cache:
            self.cache[key] = self.wrapped(path, **kwargs)
            while len(self.cache) > 2:
                self.cache.pop(next(iter(self.cache)))
        return copy.deepcopy(self.cache[key])


class DittoEngine:
    def __init__(self, settings):
        import torch

        torch.set_num_threads(4)
        sys.path.insert(0, str(ROOT / "vendor/ditto"))
        from stream_pipeline_online import StreamSDK

        cfg = settings.checkpoints / "ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
        models = settings.checkpoints / "ditto_trt_Ampere_Plus"
        if not cfg.is_file() or not models.is_dir():
            raise RuntimeError("Missing checkpoints. Run scripts/download_models.py or set DITTO_CHECKPOINTS.")
        self.sdk = StreamSDK(str(cfg), str(models))
        self.sdk.avatar_registrar = CachedRegistrar(self.sdk.avatar_registrar)
        self.active = False

    def setup(self, session):
        import torch

        torch.manual_seed(43)
        self.active = True
        self.sdk.setup(
            str(session.folder / "portrait.png"), str(session.folder / "unused.mp4"), max_size=session.resolution
        )
        self.sdk.writer.close()
        self.sdk.writer = CallbackWriter(session.on_frame)
        self.sdk.writer_pbar.close()
        self.sdk.writer_pbar = Quiet()
        self.controls(session.motion_intensity, session.mouth_intensity, initial=True)

    def controls(self, motion, mouth, initial=False):
        stitch = self.sdk.motion_stitch
        stitch.motion_intensity = motion
        stitch.mouth_strength = mouth
        if initial:
            stitch._motion_intensity = motion
            stitch._mouth_strength = mouth

    def run(self, chunk):
        self.sdk.run_chunk(np.asarray(chunk, np.float32), (3, 5, 2))
        if self.sdk.worker_exception:
            raise self.sdk.worker_exception

    def close_session(self):
        if self.active:
            try:
                self.sdk.close()
            finally:
                self.active = False
        # No model unloading or CPU offload here.
