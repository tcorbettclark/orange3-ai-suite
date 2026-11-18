"""Image Pixels widget tests: raw pixel columns and path handling."""

import numpy as np
from Orange.data import Domain, StringVariable, Table

from orangecontrib.ai_suite.widgets.ow_image_pixels import OWImagePixels


def test_pixel_values_output(qapp, capture, images_table):
    w = OWImagePixels()
    w.show()
    sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()

    out = sent[-1]
    assert out is not None
    assert len(out) == 5
    assert len(out.domain.attributes) == 100
    names = [attribute.name for attribute in out.domain.attributes]
    assert names[0] == "pixel_r0_c0"
    assert names[:3] == ["pixel_r0_c0", "pixel_r0_c1", "pixel_r0_c2"]
    assert names[10] == "pixel_r1_c0"  # the row advances once ten columns are used
    assert names[-1] == "pixel_r9_c9"
    assert out.domain.class_var.name == "class"
    assert out.domain.metas[0].name == "image"
    assert np.isfinite(out.X).all()
    assert out.X.min() >= 0.0 and out.X.max() <= 1.0
    assert out.metas[4, 0] == "disc.png"
    assert w._summary.text() == "10 x 10 = 100 values"


def test_pixel_values_dimensions_change_output(qapp, capture, images_table):
    w = OWImagePixels()
    w.show()
    sent = capture(w.Outputs.features)
    w.set_data(images_table)
    w.commit.now()
    w.wait_for_extraction()

    w.dimensions = 4
    w.commit.now()
    w.wait_for_extraction()
    assert len(sent[-1].domain.attributes) == 16
    assert w._summary.text() == "4 x 4 = 16 values"


def test_pixel_values_missing_file_skips_row(qapp, capture, image_dir):
    d, _ = image_dir
    var = StringVariable("image")
    var.attributes["type"] = "image"
    var.attributes["origin"] = str(d)
    domain = Domain([], metas=[var])
    names = [p.name for p in d.iterdir()] + ["ghost.png"]
    metas = np.asarray(names, dtype=object).reshape(-1, 1)
    table = Table.from_numpy(domain, np.empty((len(names), 0)), None, metas)

    w = OWImagePixels()
    w.show()
    sent = capture(w.Outputs.features)
    w.set_data(table)
    w.commit.now()
    w.wait_for_extraction()
    assert len(sent[-1]) == 5
    assert w.Warning.rows_skipped.is_shown()


def test_pixel_values_no_image_column(qapp):
    w = OWImagePixels()
    w.show()
    domain = Domain([], metas=[StringVariable("text")])
    table = Table.from_numpy(
        domain, np.empty((2, 0)), None, np.asarray(["a", "b"], dtype=object).reshape(-1, 1)
    )
    w.set_data(table)
    w.commit.now()
    assert w.Warning.no_image_column.is_shown()
