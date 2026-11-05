"""Hover: edge emphasis, node fill, label tint, exact restore."""

from orangecontrib.ai_suite.widgets.ow_neural_network_viewer import EdgeItem


def _edges(widget):
    return [i for i in widget.scene.items() if isinstance(i, EdgeItem)]


def test_hover_fills_nodes_and_tints_labels(widget, make_model, fire_hover):
    widget.set_model(make_model())
    edge = _edges(widget)[0]
    node, labels = edge._ends[0]
    node_rect = node.rect()
    base_colours = [label.brush().color().name() for label in labels]
    edge_colour = edge._base_pen.color().name()

    fire_hover(edge.hoverEnterEvent)
    assert node.rect() == node_rect
    assert node.brush().color().name() == edge_colour
    for label, _base in zip(labels, base_colours, strict=True):
        assert label.brush().color().name() == edge_colour
        # Hover must tint but never restyle the text itself.
        assert not label.font().bold()
    assert edge.number_label.brush().color().name() == edge_colour

    fire_hover(edge.hoverLeaveEvent)
    assert node.brush().color().name() == "#ffffff"
    for label, base in zip(labels, base_colours, strict=True):
        assert label.brush().color().name() == base
        assert not label.font().bold()
