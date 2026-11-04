"""Neural Network Viewer: draw a trained MLP with its weights annotated.

Takes the *Model* output of Orange's Neural Network widget (which wraps
scikit-learn's MLPClassifier / MLPRegressor) and renders the network on
a native Qt graphics scene: one column of nodes per layer, edges whose
width is proportional to |weight| (blue = positive, red = negative), the
numeric value printed beside each edge (for small networks), and biases
under the neurons they feed. Hovering an edge thickens it, fills its two
end nodes with its colour, and tints the associated labels with it; every
edge and node also has a tooltip with its exact value.

The diagram always scales to fit the window.
"""

import numpy as np
from AnyQt.QtCore import QLineF, QRectF, Qt
from AnyQt.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from AnyQt.QtWidgets import (
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QToolTip,
)
from Orange.base import Model
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, OWWidget


def _qt_enum(owner, group, name):
    """Resolve an enum across Qt bindings: Qt 5 exposes ``owner.name``,
    Qt 6 scopes it as ``owner.group.name``."""
    value = getattr(owner, name, None)
    if value is None:
        value = getattr(getattr(owner, group), name)
    return value


ROUND_CAP = _qt_enum(Qt, "PenCapStyle", "RoundCap")
KEEP_ASPECT = _qt_enum(Qt, "AspectRatioMode", "KeepAspectRatio")
POINTING_HAND = _qt_enum(Qt, "CursorShape", "PointingHandCursor")
NO_FRAME = _qt_enum(QGraphicsView, "Shape", "NoFrame")

# Edge-number labels are suppressed above this edge count; tooltips
# still carry every value.
MAX_LABELED_EDGES = 200
SCENE_SCALE = 600.0  # data units -> scene pixels
MARGIN = 20  # scene padding when fitting/exporting
POS_COLOR, NEG_COLOR = "#1f77b4", "#d62728"

# Z-order of scene items, bottom to top: edges < text < hovered edge
# < hovered text < node circles < text inside node circles.
Z_EDGE = 1
Z_TEXT = 2
Z_EDGE_HOVER = 2.5
Z_HOVER_TEXT = 2.6
Z_NODE = 3
Z_NODE_TEXT = 4
FILLS = {POS_COLOR: QBrush(QColor(POS_COLOR)), NEG_COLOR: QBrush(QColor(NEG_COLOR))}
EMPTY_BRUSH = QBrush(QColor("white"))
NODE_PEN = QPen(QColor("#333333"), 1.1)


def _tint(item, color=None):
    """Set a text item's colour, or restore its original (color=None)."""
    if color is None:
        item.setBrush(QBrush(item._base_color))
    else:
        item.setBrush(QBrush(QColor(color)))


def _fmt(value, spec):
    """Format a float, spelling NaN and infinities out explicitly."""
    value = float(value)
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else ("+inf" if value > 0 else "-inf")
    return format(value, spec)


class NodeItem(QGraphicsEllipseItem):
    """Accepts hover (keeping it off the edges beneath) and shows its
    tooltip instantly; it has no other hover behaviour."""

    def hoverEnterEvent(self, event):
        QToolTip.showText(event.screenPos(), self.toolTip())
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        QToolTip.hideText()
        super().hoverLeaveEvent(event)


class BoxedTextItem(QGraphicsSimpleTextItem):
    """Text with an opaque white background, for labels drawn on top
    of edges."""

    def paint(self, painter, option, widget=None):
        painter.fillRect(self.boundingRect().adjusted(-2, -1, 2, 1), QColor("white"))
        super().paint(painter, option, widget)


class EdgeItem(QGraphicsLineItem):
    """An edge whose weight is encoded in width and colour. On hover it
    thickens, its two end nodes fill with the edge's colour (size
    unchanged), and every label belonging to those nodes (names,
    numbers, biases) and to the edge itself is tinted with the edge's
    colour."""

    def __init__(self, x0, y0, x1, y1, width, color, tooltip, ends=()):
        super().__init__(QLineF(x0, y0, x1, y1))
        self._base_pen = QPen(QColor(color), width)
        self._base_pen.setCapStyle(ROUND_CAP)
        self.setPen(self._base_pen)
        self.setAcceptHoverEvents(True)
        self.setToolTip(tooltip)
        self.setCursor(POINTING_HAND)
        self.setZValue(Z_EDGE)
        self._ends = ends  # ((node, labels), (node, labels))
        self.number_label = None  # printed weight label, if shown
        self._fill = FILLS[color]

    def hoverEnterEvent(self, event):
        QToolTip.showText(event.screenPos(), self.toolTip())
        pen = QPen(self._base_pen)
        pen.setWidthF(self._base_pen.widthF() + 1.5)
        self.setPen(pen)
        self.setZValue(Z_EDGE_HOVER)
        for node, labels in self._ends:
            node.setBrush(self._fill)
            for label in labels:
                _tint(label, self._base_pen.color().name())
        if self.number_label is not None:
            _tint(self.number_label, self._base_pen.color().name())
            self.number_label.setZValue(Z_HOVER_TEXT)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        QToolTip.hideText()
        self.setPen(self._base_pen)
        self.setZValue(Z_EDGE)
        for node, labels in self._ends:
            node.setBrush(EMPTY_BRUSH)
            for label in labels:
                _tint(label)
        if self.number_label is not None:
            _tint(self.number_label)
            self.number_label.setZValue(Z_TEXT)
        super().hoverLeaveEvent(event)


class NNView(QGraphicsView):
    """Graphics view that scales the diagram to fit the window."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(NO_FRAME)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_all()

    def fit_all(self):
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN), KEEP_ASPECT)


class OWNeuralNetworkViewer(OWWidget):
    """Draw a trained neural network with weights on the edges.

    Input is the Model output of Orange's Neural Network widget (a
    scikit-learn MLP). Nodes are arranged one column per layer; edge width
    encodes |weight| and colour its sign. For small networks the numeric
    weight is printed on each edge; hovering any edge reveals exact values and
    highlights its endpoints. Biases appear under the neurons they feed and in
    tooltips.

    The view scales to fit the window. Weights that are NaN or infinite are
    drawn as zero-weight edges and flagged, and a model without recognisable
    weight matrices shows a waiting message instead of crashing.
    """

    name = "Neural Network Viewer"
    description = "Displays a trained neural network with weights on edges."
    icon = "icons/neural_network_viewer.svg"
    category = "Visualize"
    priority = 45
    keywords = ["neural network", "mlp", "weights", "visualisation"]  # noqa: RUF012

    # Class-level Setting is the canonical Orange pattern (settings are
    # stored on the class and per-instance at runtime).
    show_labels = Setting(True)

    class Inputs:
        model = Input("Model", Model)

    class Warning(OWWidget.Warning):
        not_a_network = Msg(
            "Connected model does not expose network weights. "
            "Connect the Model output of a Neural Network widget."
        )
        too_many_edges = Msg("Too many edges ({}) for numbers; hover edges for values instead.")
        save_failed = Msg("Could not write the PNG to {}.")
        non_finite = Msg(
            "Model weights contain NaN/inf values; they are drawn as zero-weight edges."
        )

    def __init__(self):
        super().__init__()
        self.model = None

        box = gui.widgetBox(self.controlArea, "Display")
        gui.checkBox(box, self, "show_labels", "Show weight numbers", callback=self.redraw)
        gui.button(box, self, "Save as PNG", callback=self.save_png)
        gui.rubber(self.controlArea)

        self.scene = QGraphicsScene(self)
        self.view = NNView(self.scene)
        self.mainArea.layout().addWidget(self.view)
        self.resize(900, 700)

    @Inputs.model
    def set_model(self, model):
        self.model = model
        self.redraw()

    # ------------------------------------------------------------------
    # Rendering

    def redraw(self):
        self.Warning.clear()
        self.scene.clear()

        coefs = intercepts = domain = None
        if self.model is not None:
            skl = getattr(self.model, "skl_model", None)
            if skl is not None:
                coefs = getattr(skl, "coefs_", None)
                intercepts = getattr(skl, "intercepts_", None)
                domain = getattr(self.model, "domain", None)

        # A usable MLP exposes one weight matrix and one bias vector per
        # layer; anything else (missing/mismatched attributes) is treated
        # as "not a network" rather than crashing _draw.
        usable = (
            coefs is not None
            and intercepts is not None
            and len(coefs) > 0
            and len(intercepts) == len(coefs)
        )
        if not usable:
            if self.model is not None:
                self.Warning.not_a_network()
            self._add_text(
                "Waiting for a neural network model…",
                SCENE_SCALE * 0.42,
                SCENE_SCALE * 0.5,
                size=12,
                color="#888888",
            )
        else:
            self._draw(coefs, intercepts, domain)

        self.view.fit_all()

    def _draw(self, coefs, intercepts, domain):
        S = SCENE_SCALE
        scene = self.scene
        sizes = [coefs[0].shape[0]] + [w.shape[1] for w in coefs]
        n_layers = len(sizes)
        in_labels, out_labels = self._layer_labels(domain, sizes)

        # Positions: one column per layer, nodes spread vertically.
        xs = np.linspace(0.15, 0.85, n_layers)
        ys = [np.linspace(0.9, 0.1, s) if s > 1 else np.array([0.5]) for s in sizes]

        def node_name(i, j):
            if i == 0:
                return in_labels[j]
            if i == n_layers - 1:
                return out_labels[j]
            return f"h{i}.{j + 1}"

        # A non-converged MLP can leave NaN/inf in its weight matrices.
        # Scale by the largest finite |weight| and flag the model; each
        # non-finite weight is clamped to zero below so the diagram is
        # never drawn with NaN geometry.
        finite_layers = [w[np.isfinite(w)] for w in coefs]
        if any(w.size != f.size for w, f in zip(coefs, finite_layers, strict=True)):
            self.Warning.non_finite()
        finite = np.concatenate(finite_layers)
        max_abs = float(np.abs(finite).max()) if finite.size else 0.0
        if max_abs <= 0:
            max_abs = 1.0
        n_edges = sum(w.size for w in coefs)
        show_numbers = self.show_labels and n_edges <= MAX_LABELED_EDGES
        if self.show_labels and n_edges > MAX_LABELED_EDGES:
            self.Warning.too_many_edges(n_edges)

        # Nodes first (edges need references to their end nodes).
        # Circle size is derived from each layer's vertical spacing, so
        # dense layers get small circles that do not overlap.
        y_gaps = [0.8 / (s - 1) if s > 1 else 0.4 for s in sizes]
        radii = [min(0.05, 0.3 * g) for g in y_gaps]
        node_items = []
        node_label_items = {}
        for i in range(n_layers):
            x = xs[i] * S
            row = []
            for j, yd in enumerate(ys[i]):
                y = yd * S
                r = radii[i] * S
                node = NodeItem(QRectF(x - r, y - r, 2 * r, 2 * r))
                node.setPen(NODE_PEN)
                node.setBrush(EMPTY_BRUSH)
                scene.addItem(node)
                node.setZValue(Z_NODE)
                # Accepting hover keeps hover events off the edges
                # underneath; the item itself has no hover behaviour.
                node.setAcceptHoverEvents(True)
                node.setCursor(POINTING_HAND)
                tooltip = node_name(i, j)
                if i > 0:
                    tooltip += f"\nbias = {_fmt(intercepts[i - 1][j], '.4g')}"
                node.setToolTip(tooltip)
                labels = []
                if i == 0:
                    labels.append(
                        self._add_text(in_labels[j], x - r - 0.012 * S, y, size=8, align="right")
                    )
                elif i == n_layers - 1:
                    labels.append(
                        self._add_text(out_labels[j], x + r + 0.012 * S, y, size=8, align="left")
                    )
                else:
                    # Number hidden neurons, but only if the circle is
                    # large enough to fit the text.
                    if r >= 0.03 * S:
                        labels.append(self._add_text(str(j + 1), x, y, size=6, z=Z_NODE_TEXT))
                if i > 0 and self.show_labels:
                    labels.append(
                        self._add_text(
                            f"b={_fmt(intercepts[i - 1][j], '.2f')}",
                            x,
                            y + r + 0.35 * r,
                            size=5.5,
                            color="#777777",
                            boxed=True,
                        )
                    )
                node_label_items[(i, j)] = labels
                row.append(node)
            node_items.append(row)

        # Edges (and their numbers).
        for i, w in enumerate(coefs):
            x0, x1 = xs[i] * S, xs[i + 1] * S
            for j in range(w.shape[0]):
                for k in range(w.shape[1]):
                    weight = float(w[j, k])
                    raw_weight = weight  # kept for the tooltip/label
                    if not np.isfinite(weight):
                        weight = 0.0  # clamp: NaN geometry would corrupt the scene
                    y0, y1 = ys[i][j] * S, ys[i + 1][k] * S
                    color = POS_COLOR if weight >= 0 else NEG_COLOR
                    tooltip = (
                        f"{node_name(i, j)} → {node_name(i + 1, k)}\n"
                        f"weight = {_fmt(raw_weight, '.4g')}"
                        + ("" if np.isfinite(raw_weight) else " (drawn as 0)")
                    )
                    edge = EdgeItem(
                        x0,
                        y0,
                        x1,
                        y1,
                        1.0 + 4.0 * abs(weight) / max_abs,
                        color,
                        tooltip,
                        ends=(
                            (node_items[i][j], node_label_items[(i, j)]),
                            (node_items[i + 1][k], node_label_items[(i + 1, k)]),
                        ),
                    )
                    scene.addItem(edge)
                    if show_numbers:
                        dx, dy = x1 - x0, y1 - y0
                        # On the line, alternating the position along it
                        # so labels on adjacent fan edges do not collide.
                        t = 0.60 if (j + k) % 2 == 0 else 0.40
                        mx = x0 + dx * t
                        my = y0 + dy * t
                        edge.number_label = self._add_text(
                            _fmt(raw_weight, ".2f"),
                            mx,
                            my,
                            size=7,
                            color=color,
                            align="center",
                            boxed=True,
                            rotation=np.degrees(np.arctan2(dy, dx)),
                        )

    def _add_text(
        self,
        text,
        x,
        y,
        size=8,
        color="#333333",
        align="center",
        rotation=None,
        boxed=False,
        z=Z_TEXT,
    ):
        item = BoxedTextItem(text) if boxed else QGraphicsSimpleTextItem(text)
        item.setZValue(z)
        font = QFont()
        font.setPointSizeF(size)
        item.setFont(font)
        item.setBrush(QBrush(QColor(color)))
        item._base_color = QColor(color)
        rect = item.boundingRect()
        dx = {"center": rect.width() / 2, "right": rect.width(), "left": 0}[align]
        item.setPos(x - dx, y - rect.height() / 2)
        if rotation is not None:
            # Rotate about the text's centre, which stays at (x, y).
            item.setTransformOriginPoint(rect.width() / 2, rect.height() / 2)
            item.setRotation(rotation)
        self.scene.addItem(item)
        return item

    # ------------------------------------------------------------------
    # Export

    def save_png(self):
        self.Warning.save_failed.clear()
        rect = self.scene.itemsBoundingRect().adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)
        if rect.isEmpty():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save network as PNG", "neural_network.png", "PNG (*.png)"
        )
        if not path:
            return
        scale = 2  # render at 2x for crisp output
        pixmap = QPixmap(int(rect.width() * scale), int(rect.height() * scale))
        pixmap.fill(QColor("white"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            self.scene.render(
                painter, QRectF(0, 0, rect.width() * scale, rect.height() * scale), rect
            )
        finally:
            painter.end()
        if not pixmap.save(path):
            self.Warning.save_failed(path)

    @staticmethod
    def _layer_labels(domain, sizes):
        n_in, n_out = sizes[0], sizes[-1]
        in_labels, out_labels = [], []
        if domain is not None:
            in_labels = [v.name for v in domain.attributes]
            class_var = domain.class_var
            if class_var is not None:
                out_labels = list(class_var.values) if class_var.is_discrete else [class_var.name]
        in_labels += [f"x{j}" for j in range(len(in_labels), n_in)]
        out_labels += [f"y{j}" for j in range(len(out_labels), n_out)]
        return in_labels[:n_in], out_labels[:n_out]


if __name__ == "__main__":  # pragma: no cover - dev preview
    from Orange.data import Table
    from Orange.regression.neural_network import NNRegressionLearner
    from orangewidget.utils.widgetpreview import WidgetPreview

    data = Table("housing")
    model = NNRegressionLearner(hidden_layer_sizes=(4,))(data)
    WidgetPreview(OWNeuralNetworkViewer).run(model)
