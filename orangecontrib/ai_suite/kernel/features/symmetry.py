"""Group F: symmetry (3 features), cheap, powerful and very visual.

Mirror symmetry features correlate one half of the image with the mirrored
other half (Pearson correlation of pixel values); the middle row/column is
ignored when the size is odd. Quadrant self-similarity is the mean of the 6
pairwise correlations between the four quadrants (each resized to the
smallest quadrant's shape, since odd sizes make them differ by one pixel).

Constant-region guard: correlation is undefined for flat inputs.
- both halves/quadrants constant and equal -> perfectly symmetric (1.0)
- both constant, different values, or exactly one constant -> 0.0
This matters for near-uniform sky images, where a NaN would poison the
feature vector.
"""

from __future__ import annotations

import cv2
import numpy as np

from .options import FeatureOptions

_EPS = 1e-12

FEATURE_NAMES = (
    "left_right_symmetry",
    "top_bottom_symmetry",
    "quadrant_self_similarity",
)

DEFINITIONS = {
    "left_right_symmetry": (
        "Bilateral symmetry: correlation of the left half with the mirrored right half."
    ),
    "top_bottom_symmetry": (
        "Vertical mirror symmetry: correlation of the top half with the mirrored bottom half."
    ),
    "quadrant_self_similarity": (
        "Uniform vs. localised structure: mean of the six pairwise inter-quadrant correlations."
    ),
}


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation resistant to float noise on constant inputs.

    Amplitude-based flat guard (std can be ~1e-8 float summation noise).
    Both sides constant and equal -> perfectly symmetric (1.0); one side
    constant, or both constant but unequal -> 0.0.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    am = a - a.mean()
    bm = b - b.mean()
    a_flat = float(np.abs(am).max()) < 1e-6
    b_flat = float(np.abs(bm).max()) < 1e-6
    if a_flat and b_flat:
        return 1.0 if abs(float(a.mean()) - float(b.mean())) < 1e-6 else 0.0
    if a_flat or b_flat:
        return 0.0
    den = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if den == 0.0 or not np.isfinite(den):
        return 0.0
    # elementwise-sum, NOT `am @ bm`: matmul dispatch hits spurious BLAS
    # warnings on some numpy builds (seen: 2.1.3/macOS arm) for identical math
    return float(np.clip((am * bm).sum() / den, -1.0, 1.0))


def _mirror_symmetry(img: np.ndarray, axis: int) -> float:
    # axis=1: left/right (split on columns); axis=0: top/bottom.
    if axis == 1:
        half = img.shape[1] // 2
        a, b = img[:, :half], img[:, img.shape[1] - half :][:, ::-1]
    else:
        half = img.shape[0] // 2
        a, b = img[:half, :], img[img.shape[0] - half :, :][::-1, :]
    if half == 0:
        return 1.0  # a single line is trivially symmetric with itself
    return _safe_corr(a, b)


def _quadrant_self_similarity(img: np.ndarray) -> float:
    h, w = img.shape
    if h < 2 or w < 2:
        return 0.0
    h2, w2 = h // 2, w // 2
    quads = [
        cv2.resize(q, (w2, h2), interpolation=cv2.INTER_AREA)
        for q in (img[:h2, :w2], img[:h2, w2:], img[h2:, :w2], img[h2:, w2:])
    ]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    corrs = [_safe_corr(quads[i], quads[j]) for i, j in pairs]
    return float(np.mean(corrs))


def compute(img: np.ndarray, opts: FeatureOptions) -> dict[str, float]:
    return {
        "left_right_symmetry": _mirror_symmetry(img, axis=1),
        "top_bottom_symmetry": _mirror_symmetry(img, axis=0),
        "quadrant_self_similarity": _quadrant_self_similarity(img),
    }
