"""Group A — global photometry."""

import numpy as np

from nn_kernel.features import FeatureOptions, photometry
from synth import uniform_image

OPTS = FeatureOptions()


def _compute(img):
    return photometry.compute(img, OPTS)


def test_symmetric_distribution_has_zero_skew():
    img = np.zeros((128, 128), dtype=np.float32)
    img[: 128 // 2] = 0.2
    img[128 // 2 :] = 0.8
    f = _compute(img)
    assert abs(f["skewness"]) < 0.05
    # two-point distribution: excess kurtosis of -2
    assert abs(f["kurtosis"] + 2.0) < 0.1


def test_bright_blobs_on_dark_positive_skew():
    img = np.zeros((128, 128), dtype=np.float32)
    img[10:20, 10:20] = 1.0  # small bright blob: few high outliers
    assert _compute(img)["skewness"] > 1.0


def test_bright_fraction_two_level_image():
    img = np.zeros((128, 128), dtype=np.float32)
    img[..., :] = 0.2
    img[:26, :] = 0.8  # 80% dark (0.2), 20% bright (0.8)
    f = _compute(img)
    # median 0.2, p10 0.2, p90 0.8 -> threshold 0.5 -> fraction ~0.2
    assert abs(f["bright_fraction"] - 0.2) < 0.02


def test_percentiles_robust_contrast():
    img = np.zeros((128, 128), dtype=np.float32)
    img[:, : 128 // 2] = 0.2
    img[:, 128 // 2 :] = 0.8
    f = _compute(img)
    assert abs(f["robust_contrast"] - 0.6) < 0.02
    assert abs(f["mean_intensity"] - 0.5) < 0.02
    assert abs(f["intensity_std"] - 0.3) < 0.02


def test_flat_image_guards():
    f = _compute(uniform_image(0.5))
    assert f["skewness"] == 0.0 and f["kurtosis"] == 0.0
    assert f["intensity_std"] == 0.0 and f["robust_contrast"] == 0.0
    assert f["bright_fraction"] == 0.0  # nothing strictly above the median
