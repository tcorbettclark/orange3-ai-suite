"""Instant tooltips on nodes and edges; labels are inert."""

from AnyQt.QtWidgets import QToolTip

from orangecontrib.ai_suite.widgets.ow_neural_network_viewer import EdgeItem, NodeItem


def _of(widget, cls):
    return [i for i in widget.scene.items() if isinstance(i, cls)]


def test_node_tooltip_is_instant(widget, make_model, fire_hover):
    widget.set_model(make_model(6, (2,)))
    node = _of(widget, NodeItem)[0]
    fire_hover(node.hoverEnterEvent)
    assert node.toolTip() in QToolTip.text()
    fire_hover(node.hoverLeaveEvent)


def test_edge_tooltip_is_instant(widget, make_model, fire_hover):
    widget.set_model(make_model(6, (2,)))
    edge = _of(widget, EdgeItem)[0]
    fire_hover(edge.hoverEnterEvent)
    assert edge.toolTip() in QToolTip.text()
    assert edge._ends[0][0].brush() == edge._fill
    fire_hover(edge.hoverLeaveEvent)


def test_labels_are_inert(widget, make_model):
    widget.set_model(make_model(6, (2,)))
    labels = [
        i
        for i in widget.scene.items()
        if type(i).__name__ in ("BoxedTextItem", "QGraphicsSimpleTextItem")
    ]
    assert all(not label.toolTip() for label in labels)
    assert all(not label.acceptHoverEvents() for label in labels)
    assert all(not label.hasCursor() for label in labels)
