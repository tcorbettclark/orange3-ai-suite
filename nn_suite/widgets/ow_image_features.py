"""Image Feature Extractor: interpretable features for image classification.

A thin shell over :mod:`nn_kernel.features`. The widget adds preprocessing and
group options, a thumbnail grid, and the Orange signal adapters.

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
from AnyQt.QtGui import QIcon, QImage, QImageReader, QPixmap
from AnyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
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


def _read_thumbnail(path: str, size: int) -> QImage:
    """Decode one image at thumbnail size, preserving aspect ratio."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid():
        reader.setScaledSize(source_size.scaled(size, size, Qt.KeepAspectRatio))
    image = reader.read()
    if image.isNull():
        return image
    return image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ThumbnailLoader(QThread):
    """Decode image thumbnails off the GUI thread, in small batches."""

    batch = pyqtSignal(int, object)  # generation, [(row, QImage), ...]

    def __init__(self, generation: int, paths: tuple[str, ...], size: int):
        super().__init__()
        self._generation = generation
        self._paths = paths
        self._size = size
        self._stop = False

    def stop(self):
        self._stop = True

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
                self.batch.emit(self._generation, pending)
                pending = []
        if pending and not self._stop:
            self.batch.emit(self._generation, pending)


class OWImageFeatureExtractor(OWWidget):
    """Extract human-interpretable features from images.

    Input is an Import Images table (image paths in a meta column). Output is
    one row per readable image, one column per selected feature, with the
    input's class variables and metas kept so any row can be traced back to its
    source image.

    The controls select preprocessing (working size, centre crop, contrast
    normalisation, optional denoising) and feature groups. Toggling groups is
    the ablation experiment; group G (colour) is the opt-in shortcut
    demonstration. The main area shows every input image as a thumbnail, with
    the count and the current selection in the header.
    """

    name = "Image Feature Extractor"
    description = "Extract interpretable image features (groups A-F) for classifiers."
    icon = "icons/image_features.svg"
    priority = 30
    keywords = ["image", "features", "interpretable", "texture", "topology"]  # noqa: RUF012

    class Inputs:
        images = Input("Images", Table)

    class Outputs:
        features = Output("Features", Table)

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
    auto_apply = Setting(True)
    working_size = Setting(128)
    centre_crop = Setting(True)
    contrast_normalise = Setting(True)
    denoise = Setting(False)
    denoise_ksize = Setting(3)
    groups_enabled = Setting("ABCDEF")
    selected_index = Setting(0)

    def __init__(self):
        super().__init__()
        self.data = None
        self._grid_paths: list[str] = []
        self._thumbnail_loader: ThumbnailLoader | None = None
        self._thumbnail_generation = 0
        self._image_cache: _ImageCache | None = None
        self._feature_cache: _FeatureCache | None = None
        self._worker: ImageExtractionWorker | None = None
        self._generation = 0
        self._data_version = 0
        self._restart_pending = False
        self._last_result: _WorkerResult | None = None
        self._progress_active = False

        prep_box = gui.widgetBox(self.controlArea, "Preprocessing")
        gui.spin(
            prep_box,
            self,
            "working_size",
            32,
            512,
            label="Working size (square)",
            callback=self._defer_commit,
        )
        gui.checkBox(
            prep_box,
            self,
            "centre_crop",
            "Centre-crop to square",
            callback=self._defer_commit,
        )
        gui.checkBox(
            prep_box,
            self,
            "contrast_normalise",
            "Contrast-normalise",
            callback=self._defer_commit,
        )
        gui.checkBox(
            prep_box,
            self,
            "denoise",
            "Denoise (median filter)",
            callback=self._defer_commit,
        )
        gui.spin(
            prep_box,
            self,
            "denoise_ksize",
            3,
            5,
            label="Denoise strength",
            callback=self._defer_commit,
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

        self._apply_button = gui.button(
            self.controlArea, self, "Extract", callback=self._commit_now
        )
        self._apply_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._auto_check = gui.checkBox(
            self.controlArea,
            self,
            "auto_apply",
            "Apply automatically",
            callback=self._auto_toggled,
        )
        self.commit.now = self._commit_now
        self.commit.deferred = self._defer_commit

        self._setup_grid()
        self.resize(950, 700)

    def _commit_now(self):
        self.commit.call()
        self.commit.dirty = False

    def _set_progress(self, value: int):
        if self._progress_active:
            self.progressBarSet(value)

    def _defer_commit(self):
        if self.auto_apply:
            self._commit_now()
        else:
            self.commit.dirty = True

    def _auto_toggled(self):
        if self.auto_apply and self.commit.dirty:
            self._commit_now()

    def _sync_group_settings(self):
        self.groups_enabled = "".join(
            letter for letter, _ in GROUP_LABELS if self._group_checks[letter].isChecked()
        )
        self._defer_commit()

    # ---- image grid --------------------------------------------------------

    def _setup_grid(self):
        container = gui.widgetBox(self.mainArea, "Images")
        header = gui.hBox(container)
        self._grid_count = QLabel("No images.")
        self._grid_selection = QLabel("")
        header.layout().addWidget(self._grid_count)
        header.layout().addStretch(1)
        header.layout().addWidget(self._grid_selection)

        self._grid = QListWidget(container)
        self._grid.setViewMode(QListView.IconMode)
        self._grid.setResizeMode(QListView.Adjust)
        self._grid.setMovement(QListView.Static)
        self._grid.setUniformItemSizes(True)
        self._grid.setGridSize(_GRID_ITEM_SIZE)
        self._grid.setIconSize(QSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE))
        self._grid.setSpacing(2)
        self._grid.setWordWrap(False)
        self._grid.setSelectionMode(QListWidget.SingleSelection)
        self._grid.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self._grid.currentRowChanged.connect(self._on_grid_row_changed)
        container.layout().addWidget(self._grid, 1)

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
            loader = ThumbnailLoader(generation, tuple(paths), _THUMBNAIL_SIZE)
            loader.batch.connect(self._on_thumbnails_loaded)
            loader.finished.connect(self._on_thumbnail_loader_finished)
            self._thumbnail_loader = loader
            loader.start()

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
        """Wait for the current thumbnail loader and deliver its batches."""
        if self._thumbnail_loader is not None:
            loader = self._thumbnail_loader
            loader.wait(timeout_ms)
            self._thumbnail_loader = None
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _feature_options(self):
        from nn_kernel.features import FeatureOptions

        return FeatureOptions(
            working_size=self.working_size,
            centre_crop=self.centre_crop,
            contrast_normalise=self.contrast_normalise,
            denoise=self.denoise,
            denoise_ksize=self.denoise_ksize,
        )

    def _preprocess_key(self) -> tuple:
        return (
            self.working_size,
            self.centre_crop,
            self.contrast_normalise,
            self.denoise,
            self.denoise_ksize,
        )

    @Inputs.images
    def set_data(self, table):
        self.data = table
        self._data_version += 1
        self._image_cache = None
        self._feature_cache = None
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
            self._set_grid_paths([])
            self.Outputs.features.send(None)
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
            self.Outputs.features.send(None)
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
            self.Outputs.features.send(None)
            return

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
        self.Outputs.features.send(out)

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

    WidgetPreview(OWImageFeatureExtractor).run()
