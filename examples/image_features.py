"""The kernel's image features and their catalogue, printed for three cloud photographs.

Run it from the repository root with ``uv run python examples/image_features.py``. The
photographs are real skies from the CCSN database (CC0), credited in
``examples/CREDITS.md``, and nothing here needs Orange or Qt.
"""

from pathlib import Path

import cv2
import numpy as np

from orangecontrib.ai_suite.kernel.images.features import (
    FeatureOptions,
    all_count_feature_names,
    all_feature_names,
    all_groups,
    extract_features,
    get_feature,
)

HERE = Path(__file__).parent


def load(name: str) -> np.ndarray:
    """Read one of the photographs in this directory as an RGB array.

    The kernel reads RGB and never guesses channel order.
    """
    bgr = cv2.imread(str(HERE / name))
    if bgr is None:
        raise SystemExit(f"cannot read {HERE / name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


CIRRUS = load("cirrus.jpg")  # wispy streaks, high cloud
VOLUTUS = load("volutus.jpg")  # a stratocumulus layer
LOW_CLOUD = load("low_cloud.jpg")  # a low cumulus layer

# 1. The catalogue is the whole feature set: one class per feature, in column
#    order, each carrying the sentence a person reads to understand the number.
print(f"catalogue   {len(all_feature_names())} features in {len(all_groups())} groups")
for group in all_groups():
    print(f"  {group.letter}  {len(group.members):2d}  {group.title}")

print("\n  every feature is a class, and its description is the documentation:")
for name in ("hole_count", "edge_density", "bluish_fraction"):
    print(f"  {name}")
    print(f"      {get_feature(name).description.summary}")
print(f"\n  count features: {', '.join(all_count_feature_names())}")

# 2. Extraction returns a FeatureVector: name -> value, plus by-group views.
#    The default selection is every group except the colour shortcut.
vec = extract_features(CIRRUS)
print(f"\ncirrus      {len(vec)} values by default (colour is opt-in)")

for group in all_groups()[:4]:
    values = vec.by_group(group.letter)[group.letter]
    shown = "  ".join(f"{name}={value:.3f}" for name, value in list(values.items())[:3])
    print(f"  {group.letter}  {group.title:18s} {shown}  ...")

print(
    f"  brightest mass at x={vec['centre_of_mass_x']:.2f} y={vec['centre_of_mass_y']:.2f}; "
    f"elongation {vec['elongation']:.2f} (1 = round, higher = streaked)"
)
print(
    f"  texture: fine {vec['fine_texture']:.3f}, coarse {vec['coarse_texture']:.3f}, "
    f"ratio {vec['coarse_to_fine_texture_ratio']:.2f}, "
    f"typical feature size {vec['autocorrelation_half_width']:.1f} px"
)

# 3. Groups select features, and nothing stops a selection being one group or
#    one feature. This is the ablation experiment the widget offers.
print("\nablation")
print(f"  groups='ABCDEFG'          {len(extract_features(CIRRUS, groups='ABCDEFG'))} columns")
print(f"  groups='D'                {len(extract_features(CIRRUS, groups='D'))} columns")
print(
    f"  exclude={{'fine_texture'}}   {len(extract_features(CIRRUS, exclude={'fine_texture'}))} columns"
)
print(
    f"  options: working_size={FeatureOptions().working_size} "
    "(every image is resized to it, which is what makes values comparable)"
)
row = extract_features(CIRRUS, groups="E")
print("  to_array() gives one Table row, in catalogue order:")
print(
    f"    { {name: round(float(value), 3) for name, value in zip(row.names(), row.to_array(), strict=True)} }"
)

# 4. Texture is where clouds separate: wispy detail lives at the fine end of the
#    scale, thick cloud at the coarse end. One ratio orders the three photos.
print("\ntexture     where the detail lives, 0.5 = scale-neutral")
print(f"  {'':10s} {'fine':>6s} {'coarse':>7s} {'ratio':>6s} {'size px':>8s}")
for label, image in (("cirrus", CIRRUS), ("volutus", VOLUTUS), ("low", LOW_CLOUD)):
    texture = extract_features(image)
    ratio = texture["coarse_to_fine_texture_ratio"]
    leaning = (
        "coarse-grained" if ratio >= 0.55 else "fine-grained" if ratio <= 0.45 else "scale-neutral"
    )
    print(
        f"  {label:10s} {texture['fine_texture']:6.3f} {texture['coarse_texture']:7.3f} "
        f"{ratio:6.2f} {texture['autocorrelation_half_width']:8.1f}  -> {leaning}"
    )

# 5. The colour shortcut, stated as data rather than as a warning: colour alone
#    separates wispy high cloud from thick low cloud, which is why the colour
#    group is off by default.
print("\nshortcut    colour features alone, off by default")
high = extract_features(CIRRUS, groups="G")
low = extract_features(LOW_CLOUD, groups="G")
for name in high.names():
    print(f"  {name:18s} high {high[name]:6.3f}   low {low[name]:6.3f}")
print(f"  -> {get_feature('bluish_fraction').description.summary}")
