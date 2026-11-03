"""Hover interception by nodes, and pointing-hand cursors."""

from orangecontrib.nn_suite.widgets.ow_nn_viewer import EdgeItem, NodeItem


def test_hover_targets_and_cursors(widget, make_model):
    widget.set_model(make_model(5, (2,)))
    nodes = [i for i in widget.scene.items() if isinstance(i, NodeItem)]
    edges = [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    assert nodes and edges
    assert all(n.acceptHoverEvents() for n in nodes)
    assert all(n.hasCursor() for n in nodes)
    assert all(e.hasCursor() for e in edges)
    for n in nodes:
        pos = n.mapToScene(n.rect().center())
        top = [i for i in widget.scene.items(pos) if i.acceptHoverEvents()]
        assert top and top[0] is n, "hover would go under the node"
