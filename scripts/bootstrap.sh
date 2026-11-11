#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${DITTO_SETUP_PYTHON:-python3.11}"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt huggingface_hub
.venv/bin/python -m pip install tensorrt_libs==8.6.1 tensorrt_bindings==8.6.1 --extra-index-url https://pypi.nvidia.com
.venv/bin/python -m pip install tensorrt==8.6.1 --no-build-isolation --no-deps
mkdir -p .cache/wheels .runtime/cudnn8
.venv/bin/python -m pip download nvidia-cudnn-cu12==8.9.7.29 --no-deps -d .cache/wheels
for wheel in .cache/wheels/nvidia_cudnn_cu12-8.9.7.29-*.whl; do
  .venv/bin/python -m zipfile -e "$wheel" .runtime/cudnn8
done
printf '\nEnvironment ready. Download models with .venv/bin/python scripts/download_models.py\n'
