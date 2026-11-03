"""Input validation: anything that is not a full MLP (coefs_ without a
matching intercepts_, or no skl_model at all) must show the warning and
the placeholder text instead of raising inside the input handler."""

import numpy as np

from orangecontrib.nn_suite.widgets.ow_nn_viewer import EdgeItem, NodeItem


class _CoefsOnly:
    """Duck-typed model exposing weights but no biases."""

    class skl_model:
        coefs_ = [np.ones((2, 2))]  # noqa: RUF012
        # no intercepts_

    domain = None


class _NoWeights:
    """Duck-typed model with an skl_model that exposes no network."""

    class skl_model:
        pass

    domain = None


class _NonFiniteWeights:
    """Duck-typed MLP whose weights and biases contain NaN and +inf."""

    class skl_model:
        coefs_ = [np.array([[np.nan, 1.0], [np.inf, -1.0]])]  # noqa: RUF012
        intercepts_ = [np.array([np.nan, np.inf])]  # noqa: RUF012

    domain = None


def _placeholder_texts(widget):
    """Only the placeholder message, not the per-node name labels (which
    are also plain QGraphicsSimpleTextItems)."""
    return [
        i
        for i in widget.scene.items()
        if type(i).__name__ == "QGraphicsSimpleTextItem"
        and "Waiting for a neural network model" in i.text()
    ]


def test_coefs_without_intercepts_warns_not_crashes(widget):
    widget.set_model(_CoefsOnly())
    assert widget.Warning.not_a_network.is_shown()
    assert not [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    assert _placeholder_texts(widget)


def test_model_without_weights_warns_not_crashes(widget):
    widget.set_model(_NoWeights())
    assert widget.Warning.not_a_network.is_shown()
    assert not [i for i in widget.scene.items() if isinstance(i, EdgeItem)]


def test_none_model_shows_placeholder_without_warning(widget):
    widget.set_model(None)
    assert not widget.Warning.not_a_network.is_shown()
    assert _placeholder_texts(widget)


def test_placeholder_disappears_once_model_connected(widget, make_model):
    widget.set_model(None)
    widget.set_model(make_model(6, (2,)))
    assert not widget.Warning.not_a_network.is_shown()
    # 6 inputs x 2 hidden + 2 hidden x 1 output
    assert len([i for i in widget.scene.items() if isinstance(i, EdgeItem)]) == 14
    assert not _placeholder_texts(widget)


def test_non_finite_weights_warn_and_clamp(widget):
    widget.set_model(_NonFiniteWeights())
    assert widget.Warning.non_finite.is_shown()
    edges = [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    assert len(edges) == 4  # 2 inputs x 2 outputs
    # NaN/inf weights are clamped to zero, so no edge gets NaN geometry.
    assert all(np.isfinite(e.pen().widthF()) for e in edges)
    # ...but the tooltips and printed labels show the true raw values.
    tooltips = {e.toolTip() for e in edges}
    assert any("weight = nan (drawn as 0)" in t for t in tooltips)
    assert any("weight = +inf (drawn as 0)" in t for t in tooltips)
    labels = [e.number_label.text() for e in edges if e.number_label]
    assert "nan" in labels and "+inf" in labels
    # Biases are not clamped (they are never used as geometry) and are
    # formatted explicitly, in both the node tooltip and the b= label.
    nodes = [i for i in widget.scene.items() if isinstance(i, NodeItem)]
    node_tooltips = "\n".join(n.toolTip() for n in nodes)
    assert "bias = nan" in node_tooltips
    assert "bias = +inf" in node_tooltips
    b_labels = [
        i.text()
        for i in widget.scene.items()
        if type(i).__name__ == "BoxedTextItem" and i.text().startswith("b=")
    ]
    assert "b=nan" in b_labels and "b=+inf" in b_labels
