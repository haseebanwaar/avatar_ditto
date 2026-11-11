#!/usr/bin/env python3
"""Configure CUDA libraries before importing torch/TensorRT, then serve one worker."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parents[1]
load_dotenv(root / ".env")
p = argparse.ArgumentParser()
p.add_argument("--host", default=os.getenv("DITTO_HOST", "127.0.0.1"))
p.add_argument("--port", type=int, default=int(os.getenv("DITTO_PORT", "7880")))
a = p.parse_args()
libs = [Path(os.getenv("DITTO_CUDNN8_DIR", str(root / ".runtime/cudnn8/nvidia/cudnn/lib")))]
for entry in sys.path:
    base = Path(entry)
    if not base.is_dir():
        continue
    libs.extend(base.glob("nvidia/*/lib"))
    libs += [base / "tensorrt_libs", base / "torch/lib"]
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in libs if x.is_dir()) + ":" + env.get("LD_LIBRARY_PATH", "")
env.setdefault("CUDA_VISIBLE_DEVICES", "0")
env.setdefault("OMP_NUM_THREADS", "4")
env.setdefault("MKL_NUM_THREADS", "4")
os.chdir(root)
os.execve(
    sys.executable,
    [sys.executable, "-m", "uvicorn", "ditto_api.app:app", "--host", a.host, "--port", str(a.port), "--workers", "1"],
    env,
)
