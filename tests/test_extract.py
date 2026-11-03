"""Registry and extraction API."""

import numpy as np
import pytest

from nn_kernel.features import (
    GROUP_ORDER,
    FeatureOptions,
    all_feature_names,
    all_features,
    extract_features,
    features_of_group,
    get_feature,
    group_names,
)
from synth import cloud_image


EXPECTED_GROUP_SIZES = {"A": 6, "B": 12, "C": 6, "D": 6, "E": 3, "F": 3}
TOTAL = sum(EXPECTED_GROUP_SIZES.values())
assert TOTAL == 36


def test_registry_structure():
    features = all_features()
    assert len(features) == TOTAL
    for letter, size in EXPECTED_GROUP_SIZES.items():
        group = features_of_group(letter)
        assert len(group) == size, letter
        assert all(f.group == letter for f in group)
    indices = [f.index for f in features]
    assert indices == list(range(TOTAL))  # stable, contiguous column order
    names = all_feature_names()
    assert len(set(names)) == TOTAL  # unique names
    assert len(set(definitions_defined(names))) == TOTAL


def definitions_defined(names):
    return [get_feature(n).definition for n in names]


def test_definitions_are_non_empty_sentences():
    for feature in all_features():
        d = feature.definition
        assert isinstance(d, str) and len(d) >= 20 and d[0].isupper(), feature.name


def test_group_names_matches_features_of_group():
    for letter in GROUP_ORDER:
        assert group_names(letter) == tuple(f.name for f in features_of_group(letter))


def test_get_feature_errors():
    with pytest.raises(ValueError, match="unknown feature"):
        get_feature("no_such_feature")
    with pytest.raises(ValueError, match="unknown feature group"):
        features_of_group("Z")


def test_extraction_full_vector():
    vec = extract_features(cloud_image(seed=1))
    assert len(vec) == TOTAL
    assert vec.names() == all_feature_names()
    arr = vec.to_array()
    assert arr.dtype == np.float64
    assert np.isfinite(arr).all()
    # deterministic
    vec2 = extract_features(cloud_image(seed=1))
    assert np.array_equal(arr, vec2.to_array())


def test_extraction_rgb_and_rgba_input():
    gray = cloud_image(seed=2)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgba = np.concatenate([rgb, np.ones_like(gray)[..., None]], axis=-1)
    base = extract_features(gray)
    assert np.allclose(extract_features(rgb).to_array(), base.to_array())
    assert np.allclose(extract_features(rgba).to_array(), base.to_array())


def test_group_subsetting_ablation():
    vec = extract_features(cloud_image(seed=1), groups="AD")
    assert vec.names() == group_names("A") + group_names("D")
    assert len(vec) == EXPECTED_GROUP_SIZES["A"] + EXPECTED_GROUP_SIZES["D"]
    assert vec.group_of("mean_intensity") == "A"
    assert vec.group_of("edge_density") == "D"


def test_feature_level_exclusion():
    full = extract_features(cloud_image(seed=1))
    fewer = extract_features(cloud_image(seed=1), exclude={"skewness", "hole_count"})
    assert len(fewer) == TOTAL - 2
    assert "skewness" not in fewer and "hole_count" not in fewer
    for name in fewer.names():
        assert fewer[name] == full[name]  # other values unchanged


def test_by_group_view():
    vec = extract_features(cloud_image(seed=1), groups="EF")
    by = vec.by_group()
    assert set(by) == {"E", "F"}
    assert set(by["E"]) == set(group_names("E"))
    assert set(by["F"]) == set(group_names("F"))


def test_error_paths():
    img = cloud_image(seed=1)
    with pytest.raises(ValueError, match="unknown feature groups"):
        extract_features(img, groups="Z")
    with pytest.raises(ValueError, match="unknown feature names"):
        extract_features(img, exclude={"not_a_feature"})
    with pytest.raises(ValueError, match="no features"):
        extract_features(img, groups="")


def test_extraction_uses_fixed_working_size():
    big = np.kron(cloud_image(seed=4), np.ones((2, 2), dtype=np.float32))  # 256x256
    vec = extract_features(big)  # centre-cropped + resized to 128
    assert len(vec) == TOTAL and np.isfinite(vec.to_array()).all()