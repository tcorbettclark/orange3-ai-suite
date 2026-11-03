"""Weight labels rotate to match their edges."""

import numpy as np

from orangecontrib.nn_suite.widgets.ow_nn_viewer import EdgeItem


def test_labels_align_with_edges(widget, make_model):
    widget.set_model(make_model())
    edges = [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    assert all(e.number_label is not None for e in edges)
    for e in edges:
        line = e.line()
        expected = np.degrees(np.arctan2(line.dy(), line.dx()))
        assert abs(e.number_label.rotation() - expected) < 1e-6
        assert -90 < e.number_label.rotation() < 90
        origin = e.number_label.transformOriginPoint()
        rect = e.number_label.boundingRect()
        assert abs(origin.x() - rect.width() / 2) < 1e-6
        assert abs(origin.y() - rect.height() / 2) < 1e-6
