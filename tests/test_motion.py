import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/ditto"))
from core.atomic_components.motion_stitch import scale_keypoint_motion


def test_motion_intensity_endpoints():
    source = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    driven = source + np.array([[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]]])
    np.testing.assert_array_equal(scale_keypoint_motion(source, driven, 0), source)
    np.testing.assert_array_equal(scale_keypoint_motion(source, driven, 1), driven)
    np.testing.assert_allclose(scale_keypoint_motion(source, driven, 0.5), (source + driven) / 2)
    assert np.linalg.norm(scale_keypoint_motion(source, driven, 1.5) - source) > np.linalg.norm(driven - source)
