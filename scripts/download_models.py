"""Download only the checkpoint files verified in the local working model bundle."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument(
    "--revision", default="main", help="Hugging Face revision; downloaded bytes must match model-manifest.json."
)
p.add_argument("--destination", type=Path, default=root / "checkpoints")
a = p.parse_args()
manifest = json.loads((root / "model-manifest.json").read_text())
for name, info in manifest.items():
    file = Path(
        hf_hub_download(
            repo_id="digital-avatar/ditto-talkinghead", filename=name, revision=a.revision, local_dir=a.destination
        )
    )
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if file.stat().st_size != info["bytes"] or digest.hexdigest() != info["sha256"]:
        raise SystemExit(
            f"Checksum mismatch for {name}. Review upstream changes; do not load unverified engines/config."
        )
    print("Verified", name)
