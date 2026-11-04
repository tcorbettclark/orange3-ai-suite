"""Image Feature Extractor widget tests: synthetic PNGs on disk."""

import numpy as np
from Orange.data import Domain, StringVariable, Table

from nn_suite.adapters import image_paths
from nn_suite.widgets.ow_image_features import OWImageFeatureExtractor

rng = np.random.default_rng(0)


def test_image_paths_adapter(images_table, image_dir):
    d, _ = image_dir
    column, paths = image_paths(images_table)
    assert column.name == "image"
    assert len(paths) == 5
    for p in paths:
        assert str(d) in p and p.endswith(".png")


def test_extraction_full_pipeline(qapp, capture, images_table, image_dir):
    from nn_kernel.features import group_names  # default = A–F, colour excluded

    expected = [n for g in ("A", "B", "C", "D", "E", "F") for n in group_names(g)]

    w = OWImageFeatureExtractor()
    w.show()
    features_sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()

    out = features_sent[-1]
    assert out is not None
    assert len(out) == 5
    assert [v.name for v in out.domain.attributes] == expected
    assert out.domain.class_var.name == "class"  # class inherited
    assert out.domain.metas[0].name == "image"  # traceability kept
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
    w.wait_for_extraction()
    out = features_sent[-1]
    assert [v.name for v in out.domain.attributes] == [
        "hole_count",
        "foreground_area_fraction",
        "stroke_crossings_median_level",
    ]
    # topology: the disc has no enclosed background -> hole_count 0; area sane
    assert out[4, "foreground_area_fraction"] > 0.05


def test_no_image_column_warning(qapp):
    w = OWImageFeatureExtractor()
    w.show()
    domain = Domain([], metas=[StringVariable("text")])
    table = Table.from_numpy(
        domain, np.empty((2, 0)), None, np.asarray(["a", "b"], dtype=object).reshape(-1, 1)
    )
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
    w.wait_for_extraction()
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
    w.wait_for_extraction()
    assert len(w._paths) == 5
    assert w._preview_spin.maximum() == 4
    w.preview_index = 4  # the disc
    w._update_preview()
    pixmap = w._preview_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    # the disc is bright at the centre; a float/uint8 channel mix-up would not be
    assert pixmap.toImage().pixelColor(98, 98).lightness() > 200
    # profiles carry data for the selected image, at the preprocessing
    # working size (128 by default), not the source resolution (64)
    assert w._row_curve.xData is not None and len(w._row_curve.xData) == 128


def test_group_toggle_auto_applies(qapp, capture, images_table):
    """The controls are wired to the commit, not just to their settings."""
    w = OWImageFeatureExtractor()
    w.show()
    sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()
    assert "bluish_fraction" not in [v.name for v in sent[-1].domain.attributes]

    w._group_checks["G"].setChecked(True)  # toggles -> settings -> commit
    w.wait_for_extraction()
    assert "bluish_fraction" in [v.name for v in sent[-1].domain.attributes]


def test_cache_reuses_features_for_groups_and_parameters(qapp, capture, images_table, monkeypatch):
    import nn_kernel.features as features_module

    calls = []
    original = features_module.extract_features

    def counted(*args, **kwargs):
        calls.append(kwargs.get("groups"))
        return original(*args, **kwargs)

    monkeypatch.setattr(features_module, "extract_features", counted)
    w = OWImageFeatureExtractor()
    w.show()
    capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()
    assert calls == ["ABCDEFG"] * 5

    # group selection is a column subset of the cached raw features
    w.groups_enabled = "E"
    w.commit.now()
    w.wait_for_extraction()
    assert calls == ["ABCDEFG"] * 5

    # parameter changes recompute only the affected profile/texture groups
    w.peak_prominence_frac = 0.2
    w.commit.now()
    w.wait_for_extraction()
    assert calls[5:] == ["B"] * 5
    w.edge_density_frac = 0.8
    w.commit.now()
    w.wait_for_extraction()
    assert calls[10:] == ["D"] * 5

    # preprocessing changes reuse the decoded images but recompute all groups
    w.working_size = 64
    w.commit.now()
    w.wait_for_extraction()
    assert calls[15:] == ["ABCDEFG"] * 5


def test_commit_controls_are_compact(qapp):
    from AnyQt.QtWidgets import QSizePolicy

    w = OWImageFeatureExtractor()
    w.show()
    assert w._commit_box.checkbox.text() == "Apply automatically"
    assert w._commit_box.button.text() == "Extract"
    assert w._commit_box.button.sizePolicy().horizontalPolicy() == QSizePolicy.Maximum
