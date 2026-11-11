from pathlib import Path
import pyximport
pyximport.install(build_dir=str(Path(__file__).resolve().parents[3]/'.cache'/'pyxbld'))
from .blend import blend_images_cy
