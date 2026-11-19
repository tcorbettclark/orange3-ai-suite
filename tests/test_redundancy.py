"""No two features may report the same number.

A pair counts as a duplicate only if it agrees to 1e-6 on *every* probe and is not
trivially near-zero everywhere. The probes are a cloud, a digit-eight, a ring, a bar,
bands and a gradient.
"""

import itertools

import numpy as np

from edu_ai_suite.images.features import all_feature_names, extract_features
from synth import band_image, cloud_image, digit8_image, ring_image, vertical_bar_image

TOL = 1e-6  # well above float32 summation noise (~1e-8), below "close by chance"


def _probes():
    gradient = np.broadcast_to(
        np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None], (96, 96)
    ).copy()
    return [
        cloud_image(seed=1),
        cloud_image(seed=4),
        digit8_image(),
        ring_image(),
        vertical_bar_image(),
        band_image(((20, 40), (70, 100))),
        gradient,
    ]


def test_no_two_features_are_duplicates():
    values = [extract_features(image, groups="ABCDEFG") for image in _probes()]
    duplicates = []
    for a, b in itertools.combinations(all_feature_names(), 2):
        agrees = all(abs(v[a] - v[b]) <= TOL * max(1.0, abs(v[a])) for v in values)
        informative = sum(abs(v[a]) > TOL for v in values) >= 2
        if agrees and informative:
            duplicates.append((a, b))
    assert not duplicates, f"duplicate feature pairs: {duplicates}"
