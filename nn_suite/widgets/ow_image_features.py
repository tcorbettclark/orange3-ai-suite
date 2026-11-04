"""Image Feature Extractor: interpretable features for image classification.

A thin shell over :mod:`nn_kernel.features`. The widget adds preprocessing and
group options, a luminance/profile preview, and the Orange signal adapters.

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
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtGui import QImage, QPixmap
from AnyQt.QtWidgets import QCheckBox, QLabel, QSizePolicy
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
    params_key: tuple


@dataclass
class _WorkerResult:
    generation: int
    data_version: int
    image_cache: _ImageCache
    feature_cache: _FeatureCache


class ImageExtractionWorker(QThread):
    """Load images and compute features without blocking the GUI thread.

    ``groups`` is the group subset to compute. ``_ALL_GROUPS`` means a full
    computation; ``"B"``/``"D"`` recompute only the parameter-dependent groups
    on top of an existing feature cache.
    """

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
        feature_cache: _FeatureCache | None,
        options,
        preprocess_key: tuple,
        params_key: tuple,
        groups: str,
    ):
        super().__init__()
        self._generation = generation
        self._data_version = data_version
        self._paths = paths
        self._skipped_no_path = skipped_no_path
        self._image_cache = image_cache
        self._feature_cache = feature_cache
        self._options = options
        self._preprocess_key = preprocess_key
        self._params_key = params_key
        self._groups = groups
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
            if self._groups == _ALL_GROUPS:
                values = np.empty((len(image_cache.images), len(names)), dtype=np.float64)
                columns = {name: i for i, name in enumerate(names)}
            else:
                names = self._feature_cache.names
                values = self._feature_cache.values.copy()
                columns = {name: i for i, name in enumerate(names)}

            total = max(1, start + len(image_cache.images))
            for n, image in enumerate(image_cache.images):
                if self._stop:
                    return
                vec = extract_features(image, self._options, groups=self._groups)
                for name, value in vec.items():
                    values[n, columns[name]] = value
                self.progress.emit(int(100 * (start + n + 1) / total))
            if not image_cache.images:
                self.progress.emit(100)

            feature_cache = _FeatureCache(
                tuple(names), values, self._preprocess_key, self._params_key
            )
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


class OWImageFeatureExtractor(OWWidget):
    """Extract human-interpretable features from images.

    Input is an Import Images table (image paths in a meta column). Output is
    one row per readable image, one column per selected feature, with the
    input's class variables and metas kept so any row can be traced back to its
    source image.

    The controls select preprocessing (working size, centre crop, contrast
    normalisation, optional denoising), peak and edge parameters, and feature
    groups. Toggling groups is the ablation experiment; group G (colour) is
    the opt-in shortcut demonstration. The preview shows the selected image's
    preprocessed luminance together with its row and column projections.
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
    peak_prominence_frac = Setting(0.10)
    edge_density_frac = Setting(0.50)
    groups_enabled = Setting("ABCDEF")
    preview_index = Setting(0)

    def __init__(self):
        super().__init__()
        self.data = None
        self._paths: list[str] = []
        self._image_cache: _ImageCache | None = None
        self._feature_cache: _FeatureCache | None = None
        self._worker: ImageExtractionWorker | None = None
        self._generation = 0
        self._data_version = 0
        self._restart_pending = False
        self._last_result: _WorkerResult | None = None

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

        param_box = gui.widgetBox(self.controlArea, "Feature parameters")
        gui.doubleSpin(
            param_box,
            self,
            "peak_prominence_frac",
            0.01,
            0.5,
            0.01,
            label="Peak prominence, of range (group B)",
            callback=self._defer_commit,
        )
        gui.doubleSpin(
            param_box,
            self,
            "edge_density_frac",
            0.1,
            1.0,
            0.05,
            label="Edge density, of gradient p99 (group D)",
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

        commit_row = gui.hBox(self.controlArea)
        self._apply_button = gui.button(commit_row, self, "Extract", callback=self._commit_now)
        self._apply_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._auto_check = gui.checkBox(
            commit_row,
            self,
            "auto_apply",
            "Apply automatically",
            callback=self._auto_toggled,
        )
        commit_row.layout().addStretch(1)
        self.commit.now = self._commit_now
        self.commit.deferred = self._defer_commit

        self._setup_preview()
        self.resize(950, 700)

    def _commit_now(self):
        self.commit.call()
        self.commit.dirty = False

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

    # ---- preview -----------------------------------------------------------

    def _setup_preview(self):
        import pyqtgraph as pg

        container = gui.widgetBox(self.mainArea, "Image preview")
        self._preview_spin = gui.spin(
            container, self, "preview_index", 0, 0, label="Image #", callback=self._update_preview
        )
        self._preview_label = QLabel("No images.")
        self._preview_label.setFixedSize(200, 200)
        self._preview_label.setAlignment(Qt.AlignCenter)
        container.layout().addWidget(self._preview_label)
        self._profiles = pg.PlotWidget(background="w")
        self._profiles.addLegend(offset=(10, 10))
        self._profiles.setLabel("bottom", "normalised position across the image")
        self._profiles.setLabel("left", "mean luminance")
        self._row_curve = self._profiles.plot(pen="#1f77b4", name="row profile")
        self._col_curve = self._profiles.plot(pen="#d62728", name="column profile")
        container.layout().addWidget(self._profiles)

    def _feature_options(self):
        from nn_kernel.features import FeatureOptions

        return FeatureOptions(
            working_size=self.working_size,
            centre_crop=self.centre_crop,
            contrast_normalise=self.contrast_normalise,
            denoise=self.denoise,
            denoise_ksize=self.denoise_ksize,
            peak_prominence_frac=self.peak_prominence_frac,
            edge_density_frac=self.edge_density_frac,
        )

    def _preprocess_key(self) -> tuple:
        return (
            self.working_size,
            self.centre_crop,
            self.contrast_normalise,
            self.denoise,
            self.denoise_ksize,
        )

    def _params_key(self) -> tuple:
        return (self.peak_prominence_frac, self.edge_density_frac)

    def _update_preview(self):
        self._preview_label.setText("No images.")
        self._row_curve.setData([], [])
        self._col_curve.setData([], [])
        if not self._paths:
            return
        try:
            import cv2

            from nn_kernel.features import preprocess

            idx = min(int(self.preview_index), len(self._paths) - 1)
            bgr = cv2.imread(self._paths[idx])
            if bgr is None:
                self._preview_label.setText("Preview unavailable.")
                return
            lum = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self._feature_options())
        except ImportError:
            return  # preview is a nicety; extraction errors carry the install hint
        h, w = lum.shape
        grayscale = np.ascontiguousarray(np.clip(lum * 255.0, 0, 255).round().astype(np.uint8))
        image = QImage(grayscale.tobytes(), w, h, w, QImage.Format_Grayscale8)
        self._preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(196, 196, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        row_profile = lum.mean(axis=1)
        col_profile = lum.mean(axis=0)
        self._row_curve.setData(np.linspace(0, 1, len(row_profile)), row_profile)
        self._col_curve.setData(np.linspace(0, 1, len(col_profile)), col_profile)

    @Inputs.images
    def set_data(self, table):
        self.data = table
        self._data_version += 1
        self._image_cache = None
        self._feature_cache = None
        self._paths = []
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
                self.progressBarFinished()
            self._paths = []
            self.Outputs.features.send(None)
            self._update_preview()
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

        kept = [(i, p) for i, p in enumerate(paths) if p]
        preprocess_key = self._preprocess_key()
        params_key = self._params_key()

        self._generation += 1
        generation = self._generation
        if self._worker is not None and self._worker.isRunning():
            self._restart_pending = True
            return

        image_cache = self._image_cache
        feature_cache = self._feature_cache
        if image_cache is None:
            groups = _ALL_GROUPS
            feature_cache = None
        elif feature_cache is None or feature_cache.preprocess_key != preprocess_key:
            groups = _ALL_GROUPS
            feature_cache = None
        elif feature_cache.params_key != params_key:
            old_peak, old_edge = feature_cache.params_key
            new_peak, new_edge = params_key
            groups = ("B" if old_peak != new_peak else "") + ("D" if old_edge != new_edge else "")
            if not groups:
                self._send_features()
                return
        else:
            self._paths = list(image_cache.paths)
            self._clamp_preview()
            self._send_features()
            return

        self._start_worker(
            generation,
            kept,
            len(paths) - len(kept),
            image_cache,
            feature_cache,
            groups,
            preprocess_key,
            params_key,
        )

    def _start_worker(
        self,
        generation,
        kept,
        skipped_no_path,
        image_cache,
        feature_cache,
        groups,
        preprocess_key,
        params_key,
    ):
        self._restart_pending = False
        worker = ImageExtractionWorker(
            generation,
            self._data_version,
            paths=tuple(kept),
            skipped_no_path=skipped_no_path,
            image_cache=image_cache,
            feature_cache=feature_cache,
            options=self._feature_options(),
            preprocess_key=preprocess_key,
            params_key=params_key,
            groups=groups,
        )
        worker.progress.connect(self.progressBarSet)
        worker.finished_ok.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        self.progressBarInit()
        worker.start()

    def _on_worker_done(self, result: _WorkerResult):
        if result is self._last_result:
            return
        self._last_result = result
        self.progressBarFinished()
        self._worker = None
        if result.data_version == self._data_version:
            # Accept a result that is still valid for the current data, even
            # when a setting changed while the worker was running.
            self._image_cache = result.image_cache
            if (
                result.feature_cache.preprocess_key == self._preprocess_key()
                and result.feature_cache.params_key == self._params_key()
            ):
                self._feature_cache = result.feature_cache
        if result.generation != self._generation:
            if self._restart_pending:
                self._restart_pending = False
                self.commit.now()
            return
        self._paths = list(result.image_cache.paths)
        self._clamp_preview()
        self._send_features()

    def _on_worker_failed(self, tb: str, generation: int):
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
            self._update_preview()
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
        self._update_preview()

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
                self.progressBarFinished()
                break

    def _clamp_preview(self):
        n = len(self._paths)
        self.preview_index = min(self.preview_index, max(n - 1, 0))
        self._preview_spin.setMaximum(max(n - 1, 0))

    def onDeleteWidget(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
            self._worker = None
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWImageFeatureExtractor).run()
