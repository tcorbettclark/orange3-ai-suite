"""Group B — profile summaries."""

import numpy as np

from nn_kernel.features import FeatureOptions, profiles
from synth import band_image, uniform_image, vertical_bar_image

OPTS = FeatureOptions()


def _compute(img, opts=OPTS):
    return profiles.compute(img, opts)


def test_two_bands_two_row_peaks():
    bands = band_image(((20, 40), (70, 100)))
    f = _compute(bands)
    assert f["row_profile_peaks"] == 2.0
    # every column crosses both bands -> flat column profile
    assert f["column_profile_peaks"] == 0.0


def test_vertical_bands_two_column_peaks():
    bands = band_image(((20, 40), (70, 100))).T.copy()
    f = _compute(bands)
    assert f["column_profile_peaks"] == 2.0
    assert f["row_profile_peaks"] == 0.0


def test_prominence_threshold_rejects_shallow_ripples():
    # Two bands plus a shallow bump of height 0.05 on a dynamic range of 1.0:
    # a 0.05 prominence shoulder must not count as a third peak, but raising
    # the sensitivity override must surface it (the "one peak with
    # shoulder vs two peaks" scale-space concern, kept as continuous policy).
    prof = np.zeros((128, 128), dtype=np.float32)
    prof[20:40] = 1.0
    prof[70:100] = 1.0
    prof[50:54] = 0.05
    assert _compute(prof)["row_profile_peaks"] == 2.0
    sensitive = _compute(prof, FeatureOptions(peak_prominence_frac=0.03))
    assert sensitive["row_profile_peaks"] == 3.0


def test_trend_sign_bottom_bright():
    bottom_bright = np.linspace(0.0, 1.0, 128, dtype=np.float32)[:, None]
    bottom_bright = np.broadcast_to(bottom_bright, (128, 128)).copy()
    f = _compute(bottom_bright)
    assert f["row_profile_trend"] > 0.9  # net change ~ +1 from top to bottom
    right_bright = bottom_bright.T.copy()
    f2 = _compute(right_bright)
    assert f2["row_profile_trend"] < 0.05  # row profile flat
    assert f2["column_profile_trend"] > 0.9


def test_flat_image_profiles():
    f = _compute(uniform_image())
    assert f["row_profile_peaks"] == 0.0
    assert f["row_profile_roughness"] == 0.0
    assert abs(f["row_profile_trend"]) < 1e-12  # polyfit float noise
    assert f["row_profile_centre_of_mass"] == 0.5
    assert f["profile_row_column_correlation"] == 0.0  # flat guard
    assert f["median_upcrossing_count"] == 0.0


def test_row_centre_of_mass_bottom_band():
    bands = band_image(((100, 128),))
    f = _compute(bands)
    # rows 100-127 bright, weighted mean row ~ 113 of 127 -> ~0.89
    assert f["row_profile_centre_of_mass"] > 0.8


def test_border_peak_counts():
    # Bright band touching the top border: the maximum of the row profile is
    # at the border and must still be counted (minimum padding).
    bands = band_image(((0, 30),))
    assert _compute(bands)["row_profile_peaks"] == 1.0


def test_median_upcrossing_vertical_bar():
    f = _compute(vertical_bar_image())
    # every row crosses the global median exactly once (into the bar);
    # columns are constant -> their own median gives no crossings
    assert abs(f["median_upcrossing_count"] - 0.5) < 1e-6


def test_diagonal_profiles_finite_and_sane():
    f = _compute(band_image(((20, 40), (70, 100))))
    for name in ("diagonal_profile_peaks", "anti_diagonal_profile_peaks"):
        assert f[name] >= 0.0
        assert float(f[name]).is_integer()
