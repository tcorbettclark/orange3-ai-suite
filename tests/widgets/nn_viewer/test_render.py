"""Pixel-level check: white label boxes mask the lines beneath.

The model is seeded so the layout is identical on every run.
"""

from AnyQt.QtCore import QRectF
from AnyQt.QtGui import QColor, QPainter, QPixmap

from nn_suite.widgets.ow_nn_viewer import EdgeItem


def test_boxes_render_white_over_lines(widget, make_model):
    widget.set_model(make_model(8, (3,)))
    edges = [i for i in widget.scene.items() if isinstance(i, EdgeItem)]
    rect = widget.scene.itemsBoundingRect()
    pixmap = QPixmap(int(rect.width()), int(rect.height()))
    pixmap.fill(QColor("white"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    widget.scene.render(painter, QRectF(0, 0, rect.width(), rect.height()), rect)
    painter.end()

    for e in edges:
        label = e.number_label
        box = label.boundingRect().adjusted(-2, -1, 2, 1)
        # A point in the box's padding band, where the line passes
        # through: the box must mask it. Boundary antialiasing can cost
        # a few intensity units (e.g. #fffefe); a line showing through
        # costs dozens (e.g. #fbecec) and fails the threshold.
        point = label.mapToScene(box.left() + 1, box.center().y())
        x = round(point.x() - rect.left())
        y = round(point.y() - rect.top())
        colour = QColor.fromRgb(pixmap.toImage().pixel(x, y))
        assert min(colour.red(), colour.green(), colour.blue()) >= 250, (
            f"box pixel {colour.name()} is not white"
        )
