"""Image Feature Extractor widget tests: synthetic PNGs on disk."""

import numpy as np
import pytest
from Orange.data import Domain, DiscreteVariable, StringVariable, Table

from nn_suite.widgets.ow_image_features import OWImageFeatureExtractor
from nn_suite.adapters import image_paths

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    """flat / noisy / disc images on disk, PNG, 64x64."""
    d = tmp_path_factory.mktemp("images")
    paths = []

    def save(arr, name):
        from PIL import Image

        path = d / name
        Image.fromarray(arr.astype(np.uint8)).save(path)
        paths.append(path)

    # flat: solid mid-gray (contrast-normalises to all-zeros)
    save(np.full((64, 64, 3), 128), "flat1.png")
    save(np.full((64, 64, 3), 130), "flat2.png")
    # noisy: strong texture
    save(rng.uniform(0, 255, size=(64, 64, 3)), "noisy1.png")
    save(rng.uniform(0, 255, size=(64, 64, 3)), "noisy2.png")
    # disc: a bright disc on black (topology-friendly)
    yy, xx = np.mgrid[0:64, 0:64]
    disc = np.zeros((64, 64, 3))
    disc[(yy - 32) ** 2 + (xx - 32) ** 2 <= 20**2] = 255
    save(disc, "disc.png")
    return d, paths


@pytest.fixture()
def images_table(image_dir):
    """Import-Images-style table: path meta marked type='image' with origin."""
    d, paths = image_dir
    var = StringVariable("image")
    var.attributes["type"] = "image"
    var.attributes["origin"] = str(d)
    domain = Domain(
        [],
        class_vars=[DiscreteVariable("class", values=("flat", "noisy", "disc"))],
        metas=[var],
    )
    names = [p.name for p in paths]
    classes = [0, 0, 1, 1, 2]
    metas = np.asarray(names, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.empty((len(names), 0)), np.asarray(classes), metas)


def test_image_paths_adapter(images_table, image_dir):
    d, _ = image_dir
    column, paths = image_paths(images_table)
    assert column.name == "image"
    assert len(paths) == 5
    for p in paths:
        assert str(d) in p and p.endswith(".png")


def test_extraction_full_pipeline(qapp, capture, images_table, image_dir):
    from nn_kernel.features import all_feature_names

    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()

    out = features_sent[-1]
    assert out is not None
    assert len(out) == 5
    assert [v.name for v in out.domain.attributes] == list(all_feature_names())
    assert out.domain.class_var.name == "class"  # class inherited
    assert out.domain.metas[0].name == "image"   # traceability kept
    assert np.isfinite(out.X).all()
    # sanity: textured images have far more gradient than the flat ones,
    # and the disc's bright topology is captured somewhere in its row
    by_name = {v.name: i for i, v in enumerate(out.domain.attributes)}
    grad = out.X[:, by_name["mean_gradient_magnitude"]]
    assert grad[2] > 10 * max(grad[0], grad[1])
    # row traceability: the meta column carries the original file names
    assert out.metas[4, 0] == "disc.png"


def test_group_ablation_via_settings(qapp, capture, images_table):
    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.groups_enabled = "E"  # topology only: hole_count, area, crossings
    w.commit.now()
    out = features_sent[-1]
    assert [v.name for v in out.domain.attributes] == [
        "hole_count", "foreground_area_fraction", "stroke_crossings_median_level",
    ]
    # topology: the disc has no enclosed background -> hole_count 0; area sane
    assert out[4, "foreground_area_fraction"] > 0.05


def test_no_image_column_warning(qapp):
    w = OWImageFeatureExtractor()
    w.show()
    domain = Domain([], metas=[StringVariable("text")])
    table = Table.from_numpy(domain, np.empty((2, 0)), None,
                             np.asarray(["a", "b"], dtype=object).reshape(-1, 1))
    w.set_data(table)
    w.commit.now()
    assert w.Warning.no_image_column.is_shown()


def test_missing_file_skips_row_with_warning(qapp, capture, images_table, image_dir, tmp_path):
    d, _ = image_dir
    var = StringVariable("image")
    var.attributes["type"] = "image"
    var.attributes["origin"] = str(d)
    domain = Domain([], metas=[var])
    names = [p.name for p in d.iterdir()] + ["ghost.png"]
    metas = np.asarray(names, dtype=object).reshape(-1, 1)
    table = Table.from_numpy(domain, np.empty((len(names), 0)), None, metas)

    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(table)
    w.commit.now()
    out = features_sent[-1]
    assert len(out) == 5
    assert w.Warning.rows_skipped.is_shown()


def test_all_groups_off(qapp, capture, images_table):
    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.groups_enabled = ""
    w.commit.now()
    assert features_sent[-1] is None
    assert w.Warning.all_groups_off.is_shown()


def test_preview_populated_after_apply(qapp, images_table):
    w = OWImageFeatureExtractor()
    w.show()
    w.set_data(images_table)
    w.commit.now()
    assert len(w._paths) == 5
    assert w._preview_spin.maximum() == 4
    w.preview_index = 4  # the disc
    w._update_preview()
    pixmap = w._preview_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    # profiles carry data for the selected image, at the preprocessing
    # working size (128 by default), not the source resolution (64)
    assert w._row_curve.xData is not None and len(w._row_curve.xData) == 128