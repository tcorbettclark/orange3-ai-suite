"""View behaviour: no zoom or pan; the diagram scales with the window."""

from AnyQt.QtCore import QSize, Qt

from nn_suite.widgets.ow_nn_viewer import EdgeItem, NNView


def test_draws_network_with_tooltips(widget, make_model):
    widget.set_model(make_model())
    edges = [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    assert len(edges) == 66
    tooltips = [i.toolTip() for i in widget.scene.items()]
    assert sum("weight =" in t for t in tooltips) >= 66
    assert sum("bias =" in t for t in tooltips) == 4


def test_view_has_no_zoom_or_pan(widget, make_model, qapp):
    widget.set_model(make_model())
    qapp.processEvents()
    view = widget.view
    # No input handler zooms the diagram.
    assert not {"wheelEvent", "mouseDoubleClickEvent"} & set(NNView.__dict__)
    # Scrollbars are hidden and the fitted content leaves no scroll
    # range, so wheel scrolling cannot pan the diagram either.
    assert view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert view.horizontalScrollBar().maximum() == 0
    assert view.verticalScrollBar().maximum() == 0


def test_scales_with_window(widget, make_model, qapp):
    widget.set_model(make_model())
    qapp.processEvents()
    initial = widget.view.transform().m11()
    widget.resize(QSize(400, 300))
    qapp.processEvents()
    shrunk = widget.view.transform().m11()
    assert abs(shrunk - initial) > 1e-6, "did not refit on resize"
    widget.resize(QSize(1200, 900))
    qapp.processEvents()
    assert widget.view.transform().m11() > shrunk


def test_redraw_refits(widget, make_model, qapp):
    widget.set_model(make_model(6, (2,)))
    widget.set_model(None)
    widget.set_model(make_model(6, (2,)))
    qapp.processEvents()


def test_save_failure_warns(widget, make_model, monkeypatch):
    widget.set_model(make_model())
    monkeypatch.setattr(
        "nn_suite.widgets.ow_nn_viewer.QFileDialog.getSaveFileName",
        lambda *a, **k: ("/nonexistent_dir_xyz/network.png", "PNG (*.png)"),
    )
    widget.save_png()
    assert widget.Warning.save_failed.is_shown()
