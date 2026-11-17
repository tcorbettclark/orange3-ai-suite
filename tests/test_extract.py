"""The feature index, the feature groups and the extraction API."""

from pathlib import Path

import numpy as np
import pytest

from orangecontrib.ai_suite.kernel.features import (
    GROUP_ORDER,
    FeatureOptions,
    all_count_feature_names,
    all_feature_names,
    all_features,
    all_groups,
    extract_features,
    features_of_group,
    get_feature,
    get_group,
)
from synth import cloud_image

EXPECTED_GROUP_SIZES = {"A": 6, "B": 10, "C": 6, "D": 6, "E": 3, "F": 3, "G": 4}
TOTAL = sum(EXPECTED_GROUP_SIZES.values())
assert TOTAL == 38


def test_index_structure():
    features = all_features()
    assert len(features) == TOTAL
    assert all_feature_names() == tuple(feature.name for feature in features)
    for letter, size in EXPECTED_GROUP_SIZES.items():
        assert len(features_of_group(letter)) == size, letter
    names = all_feature_names()
    assert len(set(names)) == TOTAL  # unique names
    assert len(set(definitions_defined(names))) == TOTAL


def definitions_defined(names):
    return [get_feature(n).definition for n in names]


def test_definitions_are_non_empty_sentences():
    for feature in all_features():
        d = feature.definition
        assert isinstance(d, str) and len(d) >= 20 and d[0].isupper(), feature.name


def test_group_members_match_features_of_group():
    for letter in GROUP_ORDER:
        assert get_group(letter).members == tuple(f.name for f in features_of_group(letter))


def test_catalogue_roster_matches_the_directory():
    from orangecontrib.ai_suite.kernel.features import catalogue

    directory = Path(catalogue.__file__).parent
    on_disk = {path.stem for path in directory.glob("*.py")} - {"__init__"}
    listed = {type(feature).__module__.rsplit(".", 1)[-1] for feature in catalogue.CATALOGUE}
    assert on_disk == listed
    assert len(catalogue.CATALOGUE) == TOTAL


def test_catalogue_entries_are_feature_instances():
    from orangecontrib.ai_suite.kernel.features import Feature, catalogue

    for feature in catalogue.CATALOGUE:
        assert isinstance(feature, Feature)


def test_group_metadata_covers_group_order():
    assert tuple(meta.letter for meta in all_groups()) == GROUP_ORDER
    for letter in GROUP_ORDER:
        meta = get_group(letter)
        assert meta.title and meta.description
        assert get_group(letter).members  # every group contributes features


def test_count_features_are_flagged():
    counts = all_count_feature_names()
    assert "hole_count" in counts
    assert "profile_median_crossing_count" in counts
    assert "mean_intensity" not in counts
    assert counts == tuple(f.name for f in all_features() if f.is_count)


def test_get_feature_errors():
    with pytest.raises(ValueError, match="unknown feature"):
        get_feature("no_such_feature")
    with pytest.raises(ValueError, match="unknown feature group"):
        features_of_group("Z")


def test_extraction_full_vector():
    vec = extract_features(cloud_image(seed=1))
    assert len(vec) == 34  # default: all groups EXCEPT the colour shortcut
    assert vec.names() == all_feature_names()[:34]  # A-F, G excluded by default
    assert all(vec.group_of(n) != "G" for n in vec.names())

    full = extract_features(cloud_image(seed=1), groups="ABCDEFG")
    assert len(full) == TOTAL
    assert np.isfinite(full.to_array()).all()
    # deterministic
    vec2 = extract_features(cloud_image(seed=1))
    assert np.array_equal(vec.to_array(), vec2.to_array())


def test_extraction_rgb_and_rgba_input():
    gray = cloud_image(seed=2)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgba = np.concatenate([rgb, np.ones_like(gray)[..., None]], axis=-1)
    base = extract_features(gray)
    assert np.allclose(extract_features(rgb).to_array(), base.to_array())
    assert np.allclose(extract_features(rgba).to_array(), base.to_array())


def test_group_subsetting_ablation():
    vec = extract_features(cloud_image(seed=1), groups="AD")
    assert vec.names() == get_group("A").members + get_group("D").members
    assert len(vec) == EXPECTED_GROUP_SIZES["A"] + EXPECTED_GROUP_SIZES["D"]
    assert vec.group_of("mean_intensity") == "A"
    assert vec.group_of("edge_density") == "D"


def test_feature_level_exclusion():
    full = extract_features(cloud_image(seed=1))
    fewer = extract_features(cloud_image(seed=1), exclude={"skewness", "hole_count"})
    assert len(fewer) == 32
    assert "skewness" not in fewer and "hole_count" not in fewer
    for name in fewer.names():
        assert fewer[name] == full[name]  # other values unchanged


def test_feature_vector_dict_protocol():
    vec = extract_features(cloud_image(seed=1), groups="E")
    assert set(vec.keys()) == set(get_group("E").members)
    assert len(list(vec)) == len(vec) == len(list(vec.items()))
    assert vec.get("hole_count") == vec["hole_count"]
    assert vec.get("not_a_feature") is None
    expected = [vec[n] for n in all_feature_names() if n in vec]
    assert vec.to_array().tolist() == expected  # stable catalogue column order


def test_by_group_view():
    vec = extract_features(cloud_image(seed=1), groups="EF")
    by = vec.by_group()
    assert set(by) == {"E", "F"}
    assert set(by["E"]) == set(get_group("E").members)
    assert set(by["F"]) == set(get_group("F").members)


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
    assert len(vec) == 34 and np.isfinite(vec.to_array()).all()


def test_one_pixel_image_yields_finite_features():
    vec = extract_features(
        np.full((1, 1), 0.5, dtype=np.float32),
        FeatureOptions(working_size=None),
    )
    assert np.isfinite(vec.to_array()).all()
