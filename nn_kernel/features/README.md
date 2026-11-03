# nn_kernel.features — feature extraction

Domain catalogue: `spec/image-feature-extractor-widget.md` §4 (groups A–F,
36 features). Group–task affinity expectation: clouds live in groups
D + B, digits in E + C + B.

## Where the documentation lives

Every feature is a first-class `Feature` object — name, group, stable column
order, and its **human-auditable definition sentence**:

```python
from nn_kernel.features import all_features, get_feature

get_feature("hole_count").definition   # "'8' -> 2, '0' -> 1, ..."
for f in all_features(): ...           # the single index, in column order
```

The definitions are canonical and live in the group modules
(``nn_kernel/features/<group>.py``, one module per group: `photometry`,
`profiles`, `spatial`, `texture`, `topology`, `symmetry`); the registry
(`registry.py`) validates names↔definitions consistency at import time and
is the single index. The docstring of each module records the exact
computation semantics (edge cases, guards, normalisations).

## Preprocessing

All features are computed on the **preprocessed luminance image** (see
`preprocess.py`): centre-cropped to a square, resized to the working size
(default 128×128), optionally median-denoised, then robustly
contrast-normalised (clip to own p1–p99, scale to [0, 1]). Colour is dropped
by design (spec §2: the "fraction of bluish pixels" shortcut must not exist
in the default feature set).

## Switching features

- group-level: `extract_features(img, groups="AD")` (ablation experiments, spec §5)
- feature-level: `extract_features(img, exclude={"skewness"})`
- parameters (peak prominence, edge-density threshold, denoise strength,
  working size) are explicit `FeatureOptions` — no hidden constants.

## Documented deviations from the spec catalogue

1. **Edge density.** The spec says "adaptive threshold"; a fixed percentile
   threshold (e.g. the gradient's own p75) returns a constant 25% for every
   image, which is degenerate. We threshold at
   `edge_density_frac × (gradient p99)` instead — adaptive yet varying.
2. **Feature count.** The spec intro says "~35 features" but its catalogue
   enumerates 36; all 36 are implemented.
3. **Border peaks.** Peak counts include maxima at the image border
   (minimum-padding); a "cloud band at the top of the frame" must count.
4. **Colour group.** Optional switchable trap group (spec §2) — deferred
   until groups A–F are tested; the registry is designed for it to slot in
   as another group module.
5. **Coarse-to-fine ratio.** The spec's ``coarse / fine`` divides two small
   numbers and is ill-conditioned exactly when images are smooth (an
   additive-noise bump legitimately halves it; stability tests exposed this).
   Implemented as the bounded, monotone equivalent
   ``coarse / (coarse + fine)`` in [0, 1] — same ranking of puffy vs. wispy,
   stable under jitter.