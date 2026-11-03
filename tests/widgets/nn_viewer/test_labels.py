"""Labels sit on their edges, staggered 40%/60% along the line."""

from collections import defaultdict

import numpy as np

from nn_suite.widgets.ow_nn_viewer import BoxedTextItem, EdgeItem


def test_labels_on_line_staggered(widget, make_model):
    widget.set_model(make_model())
    edges = [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    fans = defaultdict(list)
    for e in edges:
        label = e.number_label
        assert isinstance(label, BoxedTextItem)
        p1, p2 = e.line().p1(), e.line().p2()
        v = label.mapToScene(label.transformOriginPoint()) - p1
        d = p2 - p1
        length = np.hypot(d.x(), d.y())
        along = (v.x() * d.x() + v.y() * d.y()) / length**2
        perp = abs(v.x() * d.y() - v.y() * d.x()) / length
        assert perp < 0.5, "label is off its line"
        assert abs(along - 0.4) < 0.02 or abs(along - 0.6) < 0.02
        fans[(p1.x(), p1.y())].append((p2.y(), along))
    for fan in fans.values():
        fan.sort()
        positions = [along for _, along in fan]
        assert all(a != b for a, b in zip(positions, positions[1:])), (
            "adjacent fan labels do not alternate"
        )
