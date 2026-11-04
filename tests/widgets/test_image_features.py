"""Image Features widget tests: synthetic PNGs on disk."""

import numpy as np
from Orange.data import Domain, StringVariable, Table

from nn_suite.adapters import image_paths
from nn_suite.widgets.ow_image_features import OWImageFeatures

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

    w = OWImageFeatures()
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
    w = OWImageFeatures()
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
    w = OWImageFeatures()
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

    w = OWImageFeatures()
    w.show()
    features_sent = capture(w.Outputs.features)
    selected_sent = capture(w.Outputs.selected)
    w.set_data(table)
    w.commit.now()
    w.wait_for_extraction()
    out = features_sent[-1]
    assert len(out) == 5
    assert w.Warning.rows_skipped.is_shown()
    assert w._grid.count() == 6
    assert w._grid_count.text() == "6 images"
    w._grid.setCurrentRow(5)  # ghost.png
    assert selected_sent[-1] is None
    assert "could not be read" in w._panel_title.text()


def test_all_groups_off(qapp, capture, images_table):
    w = OWImageFeatures()
    w.show()
    features_sent = capture(w.Outputs.features)
    selected_sent = capture(w.Outputs.selected)
    w.set_data(images_table)
    w.groups_enabled = ""
    w.commit.now()
    assert features_sent[-1] is None
    assert selected_sent[-1] is None
    assert w.Warning.all_groups_off.is_shown()
    assert w._grid.count() == 5
    assert w._grid_count.text() == "5 images"


def test_grid_populated_after_apply(qapp, capture, images_table):
    w = OWImageFeatures()
    w._capture_thumbnail_batches = True
    w.show()
    features_sent = capture(w.Outputs.features)
    selected_sent = capture(w.Outputs.selected)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_thumbnails()
    w.wait_for_extraction()

    assert w._grid.count() == 5
    assert w._grid_count.text() == "5 images"
    assert [w._grid.item(i).text() for i in range(5)] == ["1", "2", "3", "4", "5"]
    assert all(not w._grid.item(i).icon().isNull() for i in range(5))
    rect = w._grid.visualRect(w._grid.model().index(0, 0))
    assert rect.width() >= 48 and rect.height() >= 48

    # the disc thumbnail is bright at the centre
    pixmap = w._grid.item(4).icon().pixmap(48, 48)
    assert pixmap.toImage().pixelColor(24, 24).lightness() > 200

    w._grid.setCurrentRow(4)  # the disc
    assert w.selected_index == 4
    assert w._grid_selection.text() == "image 5 of 5"
    assert w._panel_title.text() == "Image 5 of 5 — class disc"

    selected = selected_sent[-1]
    assert selected is not None and len(selected) == 1
    assert selected.metas[0, 0] == "disc.png"
    assert selected.ids[0] in features_sent[-1].ids

    # the panel shows values for the selected image; group G is off and greyed
    assert w._feature_groups["G"].text(0) == "G: colour (off)"
    from AnyQt.QtGui import QColor

    assert w._feature_items["bluish_fraction"].foreground(0).color() == QColor("#9a9a9a")
    w._feature_tree.setCurrentItem(w._feature_items["row_profile_peaks"])
    assert w._selected_feature == "row_profile_peaks"
    assert w._feature_items["row_profile_peaks"].text(1) != "—"
    assert "current:" in w._histogram_caption.text()
    assert len(w._histogram.getPlotItem().items) >= 1


def test_group_toggle_auto_applies(qapp, capture, images_table):
    """The controls are wired to the commit, not just to their settings."""
    w = OWImageFeatures()
    w.show()
    sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()
    assert "bluish_fraction" not in [v.name for v in sent[-1].domain.attributes]

    w._group_checks["G"].setChecked(True)  # toggles -> settings -> commit
    w.wait_for_extraction()
    assert "bluish_fraction" in [v.name for v in sent[-1].domain.attributes]


def test_cache_reuses_features_for_groups_and_preprocessing(
    qapp, capture, images_table, monkeypatch
):
    import nn_kernel.features as features_module

    calls = []
    original = features_module.extract_features

    def counted(*args, **kwargs):
        calls.append(kwargs.get("groups"))
        return original(*args, **kwargs)

    monkeypatch.setattr(features_module, "extract_features", counted)
    w = OWImageFeatures()
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

    # preprocessing changes reuse the decoded images but recompute all groups
    w.working_size = 64
    w.commit.now()
    w.wait_for_extraction()
    assert calls[5:] == ["ABCDEFG"] * 5


def test_working_size_line_edit_applies_on_enter(qapp, capture, images_table):
    from AnyQt.QtWidgets import QLineEdit

    w = OWImageFeatures()
    w.show()
    assert isinstance(w._working_size_edit, QLineEdit)
    assert w._working_size_edit.text() == "128"
    assert not hasattr(w, "_apply_button")
    assert not hasattr(w, "_auto_check")
    assert not hasattr(w, "denoise_ksize")  # fixed at the kernel default

    sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()
    assert sent[-1] is not None

    edit = w._working_size_edit
    edit.setText("64")
    edit.textEdited.emit("64")  # what typing does; setText resets the flag
    edit.returnPressed.emit()
    w.wait_for_extraction()
    assert w.working_size == 64
    assert sent[-1] is not None


def test_widget_keeps_kernel_feature_parameter_defaults(qapp):
    w = OWImageFeatures()
    w.show()
    opts = w._feature_options()
    assert opts.peak_prominence_frac == 0.10
    assert opts.edge_density_frac == 0.50


def test_wide_thumbnail_is_center_cropped(tmp_path):
    from PIL import Image

    from nn_suite.widgets.ow_image_features import _read_thumbnail

    path = tmp_path / "wide.png"
    Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8)).save(path)
    thumb = _read_thumbnail(str(path), 48)
    assert thumb.width() == 48
    assert thumb.height() == 48
