"""The z-order stack is explicit and hover-aware."""

from nn_suite.widgets import ow_nn_viewer as m
from nn_suite.widgets.ow_nn_viewer import EdgeItem, NodeItem


def _edges(widget):
    return [i for i in widget.scene.items() if isinstance(i, EdgeItem)]


def _texts(widget):
    return [
        i
        for i in widget.scene.items()
        if type(i).__name__ in ("BoxedTextItem", "QGraphicsSimpleTextItem")
    ]


def test_resting_stack(widget, make_model):
    widget.set_model(make_model(8, (3,)))
    edges = _edges(widget)
    nodes = [i for i in widget.scene.items() if isinstance(i, NodeItem)]
    assert all(e.zValue() == m.Z_EDGE for e in edges)
    assert all(n.zValue() == m.Z_NODE for n in nodes)
    in_circle = [t for t in _texts(widget) if t.zValue() == m.Z_NODE_TEXT]
    assert len(in_circle) == 3
    for t in in_circle:
        centre = t.mapToScene(t.transformOriginPoint())
        assert widget.scene.items(centre)[0] is t


def test_hover_stack(widget, make_model, fire_hover):
    widget.set_model(make_model(8, (3,)))
    edge = _edges(widget)[0]
    fire_hover(edge.hoverEnterEvent)
    assert edge.zValue() == m.Z_EDGE_HOVER
    assert edge.number_label.zValue() == m.Z_HOVER_TEXT
    assert edge.number_label.zValue() > edge.zValue() > m.Z_TEXT
    assert edge.zValue() < m.Z_NODE
    fire_hover(edge.hoverLeaveEvent)
    assert edge.zValue() == m.Z_EDGE
    assert edge.number_label.zValue() == m.Z_TEXT
