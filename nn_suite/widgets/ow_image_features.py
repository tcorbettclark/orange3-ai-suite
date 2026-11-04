"""Image Features: interpretable features for image classification.

A thin shell over :mod:`nn_kernel.features`. The widget adds preprocessing and
group options, a thumbnail grid with a feature-values panel, and the Orange
signal adapters.

Decoded images and their raw feature values are cached, so changing feature
groups reuses the cached features instead of re-reading every image. Loading and
feature computation run in a background thread; the widget's progress bar
reports both phases.
"""

from __future__ import annotations

import importlib.util
import traceback
from dataclasses import dataclass

import numpy as np
from AnyQt.QtCore import QSignalBlocker, QSize, Qt, QThread, pyqtSignal
from AnyQt.QtGui import QBrush, QColor, QIcon, QImage, QImageReader, QIntValidator, QPixmap
from AnyQt.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
)
from Orange.data import ContinuousVariable, Domain, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from nn_suite.adapters import image_paths

GROUP_LABELS = (
    ("A", "A: global photometry"),
    ("B", "B: profile summaries"),
    ("C", "C: spatial mass"),
    ("D", "D: texture"),
    ("E", "E: topology"),
    ("F", "F: symmetry"),
    ("G", "G: colour"),
)

GROUP_HELP = {
    "A": "Global photometry: brightness, contrast, skewness, kurtosis and "
    "bright fraction. Little spatial information; useful as a baseline.",
    "B": "Profile summaries: dense row, column and diagonal projections. "
    "Trend, roughness, peak counts, centre of mass and median crossings.",
    "C": "Spatial mass: where the bright matter sits, how wide or tall it is, "
    "elongation and the radial gradient.",
    "D": "Texture: gradient strength, edge density, fine and coarse texture, "
    "and autocorrelation width. Carries most signal for cloud imagery.",
    "E": "Topology: Otsu foreground, enclosed-background hole count and stroke "
    "crossings. The classic digit features.",
    "F": "Symmetry: left-right, top-bottom and quadrant self-similarity.",
    "G": "Colour: bluish fraction, blueness, saturation and greenery. A "
    "deliberate shortcut, off by default: colour alone can answer 'is there "
    "sky?' without learning any spatial structure.",
}

_ALL_GROUPS = "ABCDEFG"
_THUMBNAIL_SIZE = 48
_GRID_ITEM_SIZE = QSize(66, 76)


@dataclass
class _ImageCache:
    """Decoded RGB images for one input table."""

    paths: tuple[str, ...]
    images: list[np.ndarray]
    readable_source: tuple[int, ...]
    n_unreadable: int
    skipped_no_path: int


@dataclass
class _FeatureCache:
    """Raw feature values for every group, in ``all_feature_names()`` order."""

    names: tuple[str, ...]
    values: np.ndarray
    preprocess_key: tuple


@dataclass
class _WorkerResult:
    generation: int
    data_version: int
    image_cache: _ImageCache
    feature_cache: _FeatureCache


class ImageExtractionWorker(QThread):
    """Load images and compute all raw features without blocking the GUI."""

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(object)  # _WorkerResult
    failed = pyqtSignal(str, int)  # traceback, generation

    def __init__(
        self,
        generation: int,
        data_version: int,
        *,
        paths: tuple[str, ...],
        skipped_no_path: int,
        image_cache: _ImageCache | None,
        options,
        preprocess_key: tuple,
    ):
        super().__init__()
        self._generation = generation
        self._data_version = data_version
        self._paths = paths
        self._skipped_no_path = skipped_no_path
        self._image_cache = image_cache
        self._options = options
        self._preprocess_key = preprocess_key
        self._stop = False
        self.result: _WorkerResult | None = None
        self.error: tuple[str, int] | None = None

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import cv2

            from nn_kernel.features import all_feature_names, extract_features

            image_cache = self._image_cache
            if image_cache is None:
                images: list[np.ndarray] = []
                readable: list[int] = []
                n_unreadable = 0
                total = max(1, 2 * len(self._paths))
                for n, (source, path) in enumerate(self._paths):
                    if self._stop:
                        return
                    bgr = cv2.imread(path)
                    if bgr is None:
                        n_unreadable += 1
                    else:
                        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        readable.append(source)
                    self.progress.emit(int(100 * (n + 1) / total))
                image_cache = _ImageCache(
                    tuple(path for _, path in self._paths),
                    images,
                    tuple(readable),
                    n_unreadable,
                    self._skipped_no_path,
                )
                start = len(self._paths)
            else:
                start = 0

            names = all_feature_names()
            values = np.empty((len(image_cache.images), len(names)), dtype=np.float64)
            columns = {name: i for i, name in enumerate(names)}

            total = max(1, start + len(image_cache.images))
            for n, image in enumerate(image_cache.images):
                if self._stop:
                    return
                vec = extract_features(image, self._options, groups=_ALL_GROUPS)
                for name, value in vec.items():
                    values[n, columns[name]] = value
                self.progress.emit(int(100 * (start + n + 1) / total))
            if not image_cache.images:
                self.progress.emit(100)

            feature_cache = _FeatureCache(tuple(names), values, self._preprocess_key)
            self.result = _WorkerResult(
                self._generation,
                self._data_version,
                image_cache,
                feature_cache,
            )
            self.finished_ok.emit(self.result)
        except Exception:
            self.error = (traceback.format_exc(), self._generation)
            self.failed.emit(*self.error)


def _format_feature_value(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.4g}"


def _read_thumbnail(path: str, size: int) -> QImage:
    """Decode one image and center-crop it to a square thumbnail."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid():
        reader.setScaledSize(source_size.scaled(size, size, Qt.KeepAspectRatioByExpanding))
    image = reader.read()
    if image.isNull():
        return image
    image = image.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    x = max(0, (image.width() - size) // 2)
    y = max(0, (image.height() - size) // 2)
    return image.copy(x, y, size, size)


class ThumbnailLoader(QThread):
    """Decode image thumbnails off the GUI thread, in small batches."""

    batch = pyqtSignal(int, object)  # generation, [(row, QImage), ...]

    def __init__(self, generation: int, paths: tuple[str, ...], size: int, capture: bool = False):
        super().__init__()
        self._generation = generation
        self._paths = paths
        self._size = size
        self._capture = capture
        self.captured = [] if capture else None
        self._stop = False

    def stop(self):
        self._stop = True

    def _emit(self, pending):
        if self.captured is not None:
            self.captured.append((self._generation, pending))
        else:
            self.batch.emit(self._generation, pending)

    def run(self):
        pending = []
        for row, path in enumerate(self._paths):
            if self._stop:
                return
            if not path:
                continue
            image = _read_thumbnail(path, self._size)
            if not image.isNull():
                pending.append((row, image))
            if len(pending) >= 32:
                self._emit(pending)
                pending = []
        if pending and not self._stop:
            self._emit(pending)


class OWImageFeatures(OWWidget):
    """Extract human-interpretable features from images.

    Input is an Import Images table (image paths in a meta column). Output is
    one row per readable image, one column per selected feature, with the
    input's class variables and metas kept so any row can be traced back to its
    source image.

    The controls select preprocessing (working size, centre crop, contrast
    normalisation, optional denoising) and feature groups. Toggling groups is
    the ablation experiment; group G (colour) is the opt-in shortcut
    demonstration. The main area shows every input image as a thumbnail, with
    the count and the current selection in the header. A right-hand panel shows
    the selected image's feature values, grouped A-G, and a class-split
    histogram for the selected feature. The ``Selected image`` output sends
    that row as a one-row table; connect it to a data table's subset input to
    highlight the image among all rows.
    """

    name = "Image Features"
    description = "Extract interpretable image features (groups A-F) for classifiers."
    icon = "icons/image_features.svg"
    category = "Image Analytics"
    priority = 30
    keywords = ["image", "features", "interpretable", "texture", "topology"]  # noqa: RUF012

    class Inputs:
        images = Input("Images", Table)

    class Outputs:
        features = Output("Features", Table)
        selected = Output("Selected image", Table)

    class Warning(OWWidget.Warning):
        no_image_column = Msg("No image column found - connect the Images output of Import Images.")
        rows_skipped = Msg("{} row(s) skipped ({} unreadable, {} without a path).")
        all_groups_off = Msg("All feature groups are switched off.")

    class Error(OWWidget.Error):
        imageanalytics_missing = Msg("Orange3-ImageAnalytics is required (Options > Add-ons).")
        opencv_missing = Msg(
            "OpenCV is not available. Install the [image] extra: "
            "pip install orange3-nn-suite[image]"
        )
        extraction_failed = Msg("Feature extraction failed:\n{}")

    # settings
    working_size = Setting(128)
    centre_crop = Setting(True)
    contrast_normalise = Setting(True)
    denoise = Setting(False)
    groups_enabled = Setting("ABCDEF")
    selected_index = Setting(0)

    def __init__(self):
        super().__init__()
        self.data = None
        self._grid_paths: list[str] = []
        self._thumbnail_loader: ThumbnailLoader | None = None
        self._thumbnail_generation = 0
        self._capture_thumbnail_batches = False
        self._image_cache: _ImageCache | None = None
        self._feature_cache: _FeatureCache | None = None
        self._feature_row_by_source: dict[int, int] = {}
        self._output_table: Table | None = None
        self._selected_feature: str | None = None
        self._worker: ImageExtractionWorker | None = None
        self._generation = 0
        self._data_version = 0
        self._restart_pending = False
        self._last_result: _WorkerResult | None = None
        self._progress_active = False
        self._working_size_dirty = False

        prep_box = gui.widgetBox(self.controlArea, "Preprocessing")
        size_row = QHBoxLayout()
        prep_box.layout().addLayout(size_row)
        size_row.addWidget(QLabel("Working size (square)"))
        self._working_size_edit = QLineEdit()
        self._working_size_edit.setValidator(QIntValidator(32, 512, self))
        self._working_size_edit.setText(str(self.working_size))
        self._working_size_edit.setFixedWidth(70)
        self._working_size_edit.textEdited.connect(self._working_size_edited)
        self._working_size_edit.returnPressed.connect(self._apply_working_size)
        self._working_size_edit.editingFinished.connect(self._apply_working_size)
        size_row.addWidget(self._working_size_edit)
        size_row.addStretch(1)
        gui.checkBox(
            prep_box,
            self,
            "centre_crop",
            "Centre-crop to square",
            callback=self._apply,
        )
        gui.checkBox(
            prep_box,
            self,
            "contrast_normalise",
            "Contrast-normalise",
            callback=self._apply,
        )
        gui.checkBox(
            prep_box,
            self,
            "denoise",
            "Denoise (median filter)",
            callback=self._apply,
        )

        group_box = gui.widgetBox(self.controlArea, "Feature groups (ablation)")
        self._group_checks: dict[str, QCheckBox] = {}
        for letter, label in GROUP_LABELS:
            check = QCheckBox(label, group_box)
            check.setChecked(letter in self.groups_enabled)
            check.setToolTip(GROUP_HELP[letter])
            check.toggled.connect(self._sync_group_settings)
            group_box.layout().addWidget(check)
            self._group_checks[letter] = check

        self.controlArea.layout().addStretch(1)
        self.commit.now = self._apply
        self.commit.deferred = self._apply

        self._setup_grid()
        self.resize(1100, 750)

    def _apply(self):
        self.commit.call()

    def _working_size_edited(self, _text: str):
        self._working_size_dirty = True

    def _apply_working_size(self):
        if not self._working_size_dirty:
            return
        self._working_size_dirty = False
        try:
            value = int(self._working_size_edit.text().strip())
        except ValueError:
            self._working_size_edit.setText(str(self.working_size))
            return
        value = max(32, min(512, value))
        self._working_size_edit.setText(str(value))
        if value != self.working_size:
            self.working_size = value
            self._apply()

    def _set_progress(self, value: int):
        if self._progress_active:
            self.progressBarSet(value)

    def _sync_group_settings(self):
        self.groups_enabled = "".join(
            letter for letter, _ in GROUP_LABELS if self._group_checks[letter].isChecked()
        )
        self._refresh_panel()
        self._apply()

    # ---- image grid --------------------------------------------------------

    def _setup_grid(self):
        splitter = QSplitter(Qt.Horizontal, self.mainArea)
        splitter.setChildrenCollapsible(False)
        self.mainArea.layout().addWidget(splitter)

        grid_box = gui.widgetBox(splitter, "Images", addToLayout=False)
        splitter.addWidget(grid_box)
        header = gui.hBox(grid_box)
        self._grid_count = QLabel("No images.")
        self._grid_selection = QLabel("")
        header.layout().addWidget(self._grid_count)
        header.layout().addStretch(1)
        header.layout().addWidget(self._grid_selection)

        self._grid = QListWidget(grid_box)
        self._grid.setViewMode(QListView.IconMode)
        self._grid.setResizeMode(QListView.Adjust)
        self._grid.setMovement(QListView.Static)
        # Do not enable uniform item sizes: thumbnails arrive asynchronously,
        # and cached text-only item geometry clips the later icons.
        self._grid.setGridSize(_GRID_ITEM_SIZE)
        self._grid.setIconSize(QSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE))
        self._grid.setSpacing(2)
        self._grid.setWordWrap(False)
        self._grid.setSelectionMode(QListWidget.SingleSelection)
        self._grid.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._grid.currentRowChanged.connect(self._on_grid_row_changed)
        grid_box.layout().addWidget(self._grid, 1)

        panel = gui.widgetBox(splitter, "Feature values", addToLayout=False)
        splitter.addWidget(panel)
        splitter.setCollapsible(1, True)
        self._setup_feature_panel(panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 320])

    def _setup_feature_panel(self, panel):
        import pyqtgraph as pg

        from nn_kernel.features import features_of_group

        self._panel_title = QLabel("No image selected.")
        title_font = self._panel_title.font()
        title_font.setBold(True)
        self._panel_title.setFont(title_font)
        self._panel_title.setWordWrap(True)
        panel.layout().addWidget(self._panel_title)

        self._feature_tree = QTreeWidget()
        self._feature_tree.setHeaderLabels(["Feature", "Value"])
        self._feature_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._feature_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._feature_tree.setRootIsDecorated(True)
        self._feature_tree.setUniformRowHeights(True)
        self._feature_tree.setAlternatingRowColors(True)
        self._feature_tree.setIndentation(12)
        self._feature_tree.currentItemChanged.connect(self._on_feature_item_changed)
        panel.layout().addWidget(self._feature_tree, 1)

        self._feature_items: dict[str, QTreeWidgetItem] = {}
        self._feature_groups: dict[str, QTreeWidgetItem] = {}
        for letter, label in GROUP_LABELS:
            group_item = QTreeWidgetItem([label, ""])
            group_item.setFlags(Qt.ItemIsEnabled)
            group_font = group_item.font(0)
            group_font.setBold(True)
            group_item.setFont(0, group_font)
            self._feature_tree.addTopLevelItem(group_item)
            self._feature_groups[letter] = group_item
            for feature in features_of_group(letter):
                item = QTreeWidgetItem([feature.name, "—"])
                item.setData(0, Qt.UserRole, feature.name)
                item.setToolTip(0, feature.definition)
                item.setToolTip(1, feature.definition)
                group_item.addChild(item)
                self._feature_items[feature.name] = item
            group_item.setExpanded(True)

        self._feature_definition = QLabel("")
        self._feature_definition.setWordWrap(True)
        self._feature_definition.setTextInteractionFlags(Qt.TextSelectableByMouse)
        panel.layout().addWidget(self._feature_definition)

        self._histogram_caption = QLabel("")
        self._histogram_caption.setWordWrap(True)
        panel.layout().addWidget(self._histogram_caption)

        self._histogram = pg.PlotWidget(background="w")
        self._histogram.setMinimumHeight(170)
        self._histogram.setMenuEnabled(False)
        self._histogram.setMouseEnabled(x=False, y=False)
        self._histogram.setLabel("bottom", "value")
        self._histogram.setLabel("left", "count")
        self._histogram.addLegend(offset=(-10, 10), colCount=2, labelTextSize="8pt")
        panel.layout().addWidget(self._histogram)

    def _current_feature_cache(self):
        cache = self._feature_cache
        if cache is None or cache.preprocess_key != self._preprocess_key():
            return None
        return cache

    def _selected_feature_row(self):
        cache = self._current_feature_cache()
        if cache is None:
            return None
        return self._feature_row_by_source.get(int(self.selected_index))

    def _class_suffix(self, source: int) -> str:
        if self.data is None or not self.data.domain.class_vars:
            return ""
        var = self.data.domain.class_vars[0]
        row = self.data.Y[source]
        value = row[0] if np.ndim(row) else row
        text = var.str_val(value) if var.is_discrete else f"{float(value):.4g}"
        return f" — class {text}"

    def _refresh_panel(self):
        self._update_panel_title()
        self._update_feature_values()
        self._style_feature_groups()
        self._ensure_feature_selection()

    def _update_panel_title(self):
        n = len(self._grid_paths)
        if not n:
            self._panel_title.setText("No images.")
            return
        source = min(max(int(self.selected_index), 0), n - 1)
        prefix = f"Image {source + 1} of {n}{self._class_suffix(source)}"
        if not self._grid_paths[source]:
            self._panel_title.setText(f"{prefix} — no path.")
            return
        cache = self._current_feature_cache()
        if cache is None:
            self._panel_title.setText(prefix)
            return
        if source not in self._feature_row_by_source:
            self._panel_title.setText(f"{prefix} — image could not be read.")
            return
        self._panel_title.setText(prefix)

    def _update_feature_values(self):
        cache = self._current_feature_cache()
        row = self._selected_feature_row()
        values = {}
        if cache is not None and row is not None:
            values = {
                name: value for name, value in zip(cache.names, cache.values[row], strict=True)
            }
        for name, item in self._feature_items.items():
            item.setText(1, _format_feature_value(values.get(name)))

    def _style_feature_groups(self):
        enabled = set(self.groups_enabled)
        labels = dict(GROUP_LABELS)
        for letter, group_item in self._feature_groups.items():
            on = letter in enabled
            group_item.setText(0, labels[letter] if on else f"{labels[letter]} (off)")
            brush = QBrush(QColor("#222222" if on else "#9a9a9a"))
            group_item.setForeground(0, brush)
            group_item.setToolTip(
                0,
                labels[letter]
                if on
                else f"{labels[letter]} — switched off, so not in the Features output.",
            )
            for i in range(group_item.childCount()):
                child = group_item.child(i)
                child.setForeground(0, brush)
                child.setForeground(1, brush)

    def _ensure_feature_selection(self):
        if not self._feature_items:
            self._selected_feature = None
            self._update_feature_details()
            return
        if self._selected_feature not in self._feature_items:
            self._selected_feature = None
            for letter, _ in GROUP_LABELS:
                if letter not in self.groups_enabled:
                    continue
                group_item = self._feature_groups[letter]
                if group_item.childCount():
                    self._selected_feature = group_item.child(0).data(0, Qt.UserRole)
                    break
            if self._selected_feature is None:
                self._selected_feature = (
                    self._feature_tree.topLevelItem(0).child(0).data(0, Qt.UserRole)
                )
        item = self._feature_items[self._selected_feature]
        with QSignalBlocker(self._feature_tree):
            self._feature_tree.setCurrentItem(item)
        self._update_feature_details()

    def _on_feature_item_changed(self, current, _previous):
        if current is None:
            return
        name = current.data(0, Qt.UserRole)
        if not name:
            return
        self._selected_feature = name
        self._update_feature_details()

    def _update_feature_details(self):
        name = self._selected_feature
        if name is None:
            self._feature_definition.setText("")
            self._histogram_caption.setText("")
            self._clear_histogram()
            return
        from nn_kernel.features import get_feature

        self._feature_definition.setText(get_feature(name).definition)
        self._update_histogram(name)

    def _clear_histogram(self):
        plot = self._histogram.getPlotItem()
        plot.clear()
        if plot.legend is not None:
            plot.legend.clear()

    def _update_histogram(self, feature_name: str):
        import pyqtgraph as pg

        from nn_kernel.features import get_feature

        plot = self._histogram.getPlotItem()
        self._clear_histogram()
        cache = self._current_feature_cache()
        row = self._selected_feature_row()
        if cache is None or row is None:
            self._histogram_caption.setText("")
            return
        try:
            column = cache.names.index(feature_name)
        except ValueError:
            self._histogram_caption.setText("This feature was not extracted.")
            return

        values = np.asarray(cache.values[:, column], dtype=float)
        sources = np.asarray(self._image_cache.readable_source, dtype=int)
        class_var = self.data.domain.class_var if self.data is not None else None
        labels = None
        if class_var is not None and class_var.is_discrete:
            labels = self.data.Y[sources]
            if labels.ndim == 2:
                labels = labels[:, 0]
        finite = np.isfinite(values)
        values = values[finite]
        if labels is not None:
            labels = labels[finite]
        current = float(cache.values[row, column])
        if not values.size:
            self._histogram_caption.setText("No finite values to plot.")
            return

        vmin, vmax = float(values.min()), float(values.max())
        if vmin == vmax:
            edges = np.array([vmin - 0.5, vmax + 0.5])
        elif np.allclose(values, np.round(values)) and vmax - vmin <= 60:
            lo, hi = int(np.floor(vmin)), int(np.ceil(vmax))
            edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
        else:
            edges = np.linspace(vmin, vmax, 21)
        centres = (edges[:-1] + edges[1:]) / 2
        width = np.diff(edges)

        if labels is None:
            series = [("all images", None)]
        else:
            series = [(class_var.str_val(value), labels == value) for value in np.unique(labels)]

        bottom = np.zeros(len(centres))
        for index, (label, mask) in enumerate(series):
            counts, _ = np.histogram(values if mask is None else values[mask], bins=edges)
            if not counts.any():
                continue
            color = pg.intColor(index, hues=max(3, len(series)))
            bar = pg.BarGraphItem(
                x=centres,
                height=counts,
                width=width * 0.9,
                bottom=bottom,
                brush=color,
                pen=pg.mkPen(color),
            )
            plot.addItem(bar)
            if plot.legend is not None:
                plot.legend.addItem(bar, label)
            bottom += counts

        if np.isfinite(current):
            plot.addItem(
                pg.InfiniteLine(
                    pos=current,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen("#d62728", width=2),
                )
            )
        caption = f"current: {_format_feature_value(current)}"
        if labels is not None:
            caption += " — bars by class"
        if get_feature(feature_name).group not in self.groups_enabled:
            caption += " — group off (not in output)"
        self._histogram_caption.setText(caption)

    def _set_grid_paths(self, paths):
        paths = list(paths)
        self._grid_paths = paths
        self._thumbnail_generation += 1
        generation = self._thumbnail_generation
        if self._thumbnail_loader is not None:
            self._thumbnail_loader.stop()
            self._thumbnail_loader.wait()
            self._thumbnail_loader = None

        with QSignalBlocker(self._grid):
            self._grid.clear()
            for i, path in enumerate(paths):
                item = QListWidgetItem(str(i + 1))
                item.setToolTip(path or "No path")
                self._grid.addItem(item)

        n_with_path = sum(1 for path in paths if path)
        missing = len(paths) - n_with_path
        if n_with_path:
            label = f"{n_with_path} image{'s' if n_with_path != 1 else ''}"
            if missing:
                label += f" ({missing} without a path)"
            self._grid_count.setText(label)
        else:
            self._grid_count.setText("No images.")
        self._restore_grid_selection()

        if n_with_path:
            loader = ThumbnailLoader(
                generation,
                tuple(paths),
                _THUMBNAIL_SIZE,
                capture=self._capture_thumbnail_batches,
            )
            loader.batch.connect(self._on_thumbnails_loaded)
            loader.finished.connect(self._on_thumbnail_loader_finished)
            self._thumbnail_loader = loader
            loader.start()
        self._refresh_panel()
        self._send_selected()

    def _restore_grid_selection(self):
        n = self._grid.count()
        if not n:
            self._grid_selection.setText("")
            return
        row = min(max(int(self.selected_index), 0), n - 1)
        self.selected_index = row
        with QSignalBlocker(self._grid):
            self._grid.setCurrentRow(row)
            self._grid.scrollToItem(self._grid.item(row))
        self._grid_selection.setText(f"image {row + 1} of {n}")

    def _on_grid_row_changed(self, row: int):
        if row < 0:
            return
        self.selected_index = row
        self._grid_selection.setText(f"image {row + 1} of {self._grid.count()}")
        self._send_selected()
        self._refresh_panel()

    def _on_thumbnails_loaded(self, generation: int, batch):
        if generation != self._thumbnail_generation:
            return
        for row, image in batch:
            item = self._grid.item(row)
            if item is not None and not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image)))

    def _on_thumbnail_loader_finished(self):
        if self.sender() is self._thumbnail_loader:
            self._thumbnail_loader = None

    def wait_for_thumbnails(self, timeout_ms: int = 30000):
        """Wait for thumbnails; set ``_capture_thumbnail_batches`` first in tests."""
        if self._thumbnail_loader is not None:
            loader = self._thumbnail_loader
            loader.wait(timeout_ms)
            self._thumbnail_loader = None
            if loader.captured is not None:
                for generation, batch in loader.captured:
                    self._on_thumbnails_loaded(generation, batch)

    def _feature_options(self):
        from nn_kernel.features import FeatureOptions

        return FeatureOptions(
            working_size=self.working_size,
            centre_crop=self.centre_crop,
            contrast_normalise=self.contrast_normalise,
            denoise=self.denoise,
        )

    def _preprocess_key(self) -> tuple:
        return (
            self.working_size,
            self.centre_crop,
            self.contrast_normalise,
            self.denoise,
        )

    @Inputs.images
    def set_data(self, table):
        self.data = table
        self._data_version += 1
        self._image_cache = None
        self._feature_cache = None
        self._feature_row_by_source = {}
        self._output_table = None
        paths = []
        if table is not None:
            try:
                _, paths = image_paths(table)
            except ValueError:
                pass
        self._set_grid_paths(paths)
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        self.Warning.no_image_column.clear()
        self.Warning.rows_skipped.clear()
        self.Warning.all_groups_off.clear()
        self.Error.imageanalytics_missing.clear()
        self.Error.opencv_missing.clear()
        self.Error.extraction_failed.clear()
        if self.data is None:
            self._generation += 1
            self._restart_pending = False
            if self._worker is not None:
                self._worker.stop()
                self._worker.wait()
                self._worker = None
                self._progress_active = False
                self.progressBarFinished()
            self._feature_row_by_source = {}
            self._output_table = None
            self._set_grid_paths([])
            self.Outputs.features.send(None)
            self.Outputs.selected.send(None)
            return
        try:
            self._extract_and_send()
        except ValueError as exc:
            message = str(exc)
            if "ImageAnalytics" in message:
                self.Error.imageanalytics_missing()
            elif "no image column" in message:
                self.Warning.no_image_column(message)
            elif "OpenCV" in message or "cv2" in message:
                self.Error.opencv_missing()
            else:
                self.Error.extraction_failed(message)
        except Exception as exc:
            if isinstance(exc, ModuleNotFoundError) and "cv2" in str(exc):
                self.Error.opencv_missing()
            else:
                self.Error.extraction_failed(f"{type(exc).__name__}: {exc}")

    def _extract_and_send(self):
        if importlib.util.find_spec("cv2") is None:
            raise ValueError("OpenCV not available - install the [image] extra")
        if not self.groups_enabled:
            self.Warning.all_groups_off()
            self._output_table = None
            self._feature_row_by_source = {}
            self.Outputs.features.send(None)
            self.Outputs.selected.send(None)
            self._refresh_panel()
            return
        _, paths = image_paths(self.data)
        if list(paths) != self._grid_paths:
            self._set_grid_paths(paths)

        kept = [(i, p) for i, p in enumerate(paths) if p]
        preprocess_key = self._preprocess_key()

        self._generation += 1
        generation = self._generation
        if self._worker is not None and self._worker.isRunning():
            self._restart_pending = True
            return

        image_cache = self._image_cache
        feature_cache = self._feature_cache
        if (
            image_cache is not None
            and feature_cache is not None
            and feature_cache.preprocess_key == preprocess_key
        ):
            self._send_features()
            return

        self._start_worker(
            generation,
            kept,
            len(paths) - len(kept),
            image_cache,
            preprocess_key,
        )

    def _start_worker(
        self,
        generation,
        kept,
        skipped_no_path,
        image_cache,
        preprocess_key,
    ):
        self._restart_pending = False
        worker = ImageExtractionWorker(
            generation,
            self._data_version,
            paths=tuple(kept),
            skipped_no_path=skipped_no_path,
            image_cache=image_cache,
            options=self._feature_options(),
            preprocess_key=preprocess_key,
        )
        worker.progress.connect(self._set_progress)
        worker.finished_ok.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        self.progressBarInit()
        self._progress_active = True
        worker.start()
        self._refresh_panel()

    def _on_worker_done(self, result: _WorkerResult):
        if result is self._last_result:
            return
        self._last_result = result
        self._progress_active = False
        self.progressBarFinished()
        self._worker = None
        if result.data_version == self._data_version:
            # Accept a result that is still valid for the current data, even
            # when a setting changed while the worker was running.
            self._image_cache = result.image_cache
            if result.feature_cache.preprocess_key == self._preprocess_key():
                self._feature_cache = result.feature_cache
        if result.generation != self._generation:
            if self._restart_pending:
                self._restart_pending = False
                self.commit.now()
            return
        self._send_features()

    def _on_worker_failed(self, tb: str, generation: int):
        self._progress_active = False
        self.progressBarFinished()
        self._worker = None
        if generation != self._generation:
            if self._restart_pending:
                self._restart_pending = False
                self.commit.now()
            return
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        self.Error.extraction_failed(last)
        self._refresh_panel()
        self._send_selected()

    def _send_features(self):
        from nn_kernel.features import group_names

        image_cache = self._image_cache
        feature_cache = self._feature_cache
        self.Warning.rows_skipped(
            image_cache.n_unreadable + image_cache.skipped_no_path,
            image_cache.n_unreadable,
            image_cache.skipped_no_path,
            shown=bool(image_cache.n_unreadable or image_cache.skipped_no_path),
        )
        if not feature_cache.values.size:
            self._feature_row_by_source = {}
            self._output_table = None
            self.Outputs.features.send(None)
            self.Outputs.selected.send(None)
            self._refresh_panel()
            return

        self._feature_row_by_source = {
            int(source): i for i, source in enumerate(image_cache.readable_source)
        }
        ordered = [
            name
            for letter, _ in GROUP_LABELS
            if letter in self.groups_enabled
            for name in group_names(letter)
        ]
        columns = {name: i for i, name in enumerate(feature_cache.names)}
        values = feature_cache.values[:, [columns[name] for name in ordered]]
        source = self.data.domain
        kept_idx = np.asarray(image_cache.readable_source, dtype=int)
        domain = Domain(
            [ContinuousVariable(name) for name in ordered],
            class_vars=list(source.class_vars),
            metas=list(source.metas),
        )
        out = Table.from_numpy(
            domain,
            values,
            self.data.Y[kept_idx] if source.class_vars else None,
            self.data.metas[kept_idx],
        )
        self._output_table = out
        self.Outputs.features.send(out)
        self._send_selected()
        self._refresh_panel()

    def _send_selected(self):
        table = self._output_table
        row = self._feature_row_by_source.get(int(self.selected_index))
        if table is None or row is None or row >= len(table):
            self.Outputs.selected.send(None)
            return
        self.Outputs.selected.send(table[row : row + 1])

    def wait_for_extraction(self, timeout_ms: int = 30000):
        """Wait for the current and any pending extraction worker.

        The worker stores its result before emitting the signal, so tests can
        consume it without spinning the GUI event loop.
        """
        while self._worker is not None:
            worker = self._worker
            worker.wait(timeout_ms)
            self._worker = None
            if worker.result is not None:
                self._on_worker_done(worker.result)
            elif worker.error is not None:
                self._on_worker_failed(*worker.error)
            else:
                self._progress_active = False
                self.progressBarFinished()
                break

    def onDeleteWidget(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
            self._worker = None
        if self._thumbnail_loader is not None:
            self._thumbnail_loader.stop()
            self._thumbnail_loader.wait()
            self._thumbnail_loader = None
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWImageFeatures).run()
