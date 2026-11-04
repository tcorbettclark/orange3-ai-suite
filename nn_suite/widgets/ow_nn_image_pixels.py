"""NN Image Pixels: flatten downsampled images into raw pixel columns.

A deliberately vanilla baseline: convert each image to luminance, resize it to
a square grid and flatten it row-major. There is no convolution, no spatial
encoding and no feature engineering - just the pixels.
"""

from __future__ import annotations

import importlib.util
import traceback
from dataclasses import dataclass

import numpy as np
from AnyQt.QtCore import QThread, pyqtSignal
from AnyQt.QtWidgets import QLabel
from Orange.data import ContinuousVariable, Domain, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from nn_suite.adapters import image_paths


@dataclass
class _PixelResult:
    generation: int
    data_version: int
    values: np.ndarray
    readable_source: tuple[int, ...]
    n_unreadable: int
    skipped_no_path: int
    size: int


class PixelFeatureWorker(QThread):
    """Load and flatten images without blocking the GUI thread."""

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(object)  # _PixelResult
    failed = pyqtSignal(str, int)  # traceback, generation

    def __init__(
        self,
        generation: int,
        data_version: int,
        *,
        paths: tuple[tuple[int, str], ...],
        size: int,
        skipped_no_path: int,
    ):
        super().__init__()
        self._generation = generation
        self._data_version = data_version
        self._paths = paths
        self._size = size
        self._skipped_no_path = skipped_no_path
        self._stop = False
        self.result: _PixelResult | None = None
        self.error: tuple[str, int] | None = None

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import cv2

            from nn_kernel.pixels import pixel_features

            rows = []
            readable = []
            n_unreadable = 0
            total = max(1, len(self._paths))
            for n, (source, path) in enumerate(self._paths):
                if self._stop:
                    return
                bgr = cv2.imread(path)
                if bgr is None:
                    n_unreadable += 1
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rows.append(pixel_features(rgb, self._size))
                    readable.append(source)
                self.progress.emit(int(100 * (n + 1) / total))

            values = (
                np.vstack(rows)
                if rows
                else np.empty((0, self._size * self._size), dtype=np.float32)
            )
            self.result = _PixelResult(
                self._generation,
                self._data_version,
                values,
                tuple(readable),
                n_unreadable,
                self._skipped_no_path,
                self._size,
            )
            self.finished_ok.emit(self.result)
        except Exception:
            self.error = (traceback.format_exc(), self._generation)
            self.failed.emit(*self.error)


class OWNNImagePixels(OWWidget):
    """Flatten each input image into raw pixel features.

    Input is an Import Images table. Every image is converted to luminance,
    resized to a square grid and flattened row-major into values in [0, 1].
    The only control is the grid dimension; the default 10 produces 100
    features per image.

    The output keeps the input's class variables and meta columns, including
    the image path, so rows remain traceable. This widget is intentionally
    plain: it is the no-convolution, no-feature-engineering baseline.
    """

    name = "NN Image Pixels"
    description = "Flatten downsampled images into raw pixel features."
    icon = "icons/nn_image_pixels.svg"
    priority = 35
    keywords = ["pixels", "raw", "image", "baseline", "flatten"]  # noqa: RUF012

    class Inputs:
        images = Input("Images", Table)

    class Outputs:
        features = Output("Features", Table)

    class Warning(OWWidget.Warning):
        no_image_column = Msg("No image column found - connect the Images output of Import Images.")
        rows_skipped = Msg("{} row(s) skipped ({} unreadable, {} without a path).")
        no_images = Msg("No readable images found.")

    class Error(OWWidget.Error):
        imageanalytics_missing = Msg("Orange3-ImageAnalytics is required (Options > Add-ons).")
        opencv_missing = Msg(
            "OpenCV is not available. Install the [image] extra: "
            "pip install orange3-nn-suite[image]"
        )
        extraction_failed = Msg("Pixel extraction failed:\n{}")

    dimensions = Setting(10)

    def __init__(self):
        super().__init__()
        self.data = None
        self._worker: PixelFeatureWorker | None = None
        self._generation = 0
        self._data_version = 0
        self._restart_pending = False
        self._last_result: _PixelResult | None = None
        self._progress_active = False

        box = gui.widgetBox(self.controlArea, "Pixels")
        self._size_spin = gui.spin(
            box,
            self,
            "dimensions",
            2,
            64,
            label="Dimensions (square)",
            callback=self._apply,
        )
        self._summary = QLabel()
        box.layout().addWidget(self._summary)
        self.controlArea.layout().addStretch(1)
        self._refresh_summary()

        self.commit.now = self._apply
        self.commit.deferred = self._apply

        explanation = QLabel(
            "Each image is converted to luminance, resized to a square grid and "
            "flattened row-major into values in [0, 1]. No spatial encoding is added."
        )
        explanation.setWordWrap(True)
        self.mainArea.layout().addWidget(explanation)
        self.mainArea.layout().addStretch(1)
        self.resize(520, 280)

    def _apply(self):
        self._refresh_summary()
        self.commit.call()

    def _refresh_summary(self):
        self._summary.setText(
            f"{self.dimensions} x {self.dimensions} = {self.dimensions * self.dimensions} features"
        )

    @Inputs.images
    def set_data(self, table):
        self.data = table
        self._data_version += 1
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        self.Warning.no_image_column.clear()
        self.Warning.rows_skipped.clear()
        self.Warning.no_images.clear()
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
            self.Outputs.features.send(None)
            return
        try:
            self._extract_and_send()
        except ValueError as exc:
            self.Outputs.features.send(None)
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
            self.Outputs.features.send(None)
            if isinstance(exc, ModuleNotFoundError) and "cv2" in str(exc):
                self.Error.opencv_missing()
            else:
                self.Error.extraction_failed(f"{type(exc).__name__}: {exc}")

    def _extract_and_send(self):
        if importlib.util.find_spec("cv2") is None:
            raise ValueError("OpenCV not available - install the [image] extra")
        _, paths = image_paths(self.data)
        kept = tuple((i, path) for i, path in enumerate(paths) if path)
        self._generation += 1
        generation = self._generation
        if self._worker is not None and self._worker.isRunning():
            self._restart_pending = True
            return
        self._start_worker(generation, kept, len(paths) - len(kept))

    def _start_worker(self, generation, kept, skipped_no_path):
        self._restart_pending = False
        worker = PixelFeatureWorker(
            generation,
            self._data_version,
            paths=kept,
            size=self.dimensions,
            skipped_no_path=skipped_no_path,
        )
        worker.progress.connect(self._set_progress)
        worker.finished_ok.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._worker = worker
        self.progressBarInit()
        self._progress_active = True
        worker.start()

    def _set_progress(self, value: int):
        if self._progress_active:
            self.progressBarSet(value)

    def _on_worker_done(self, result: _PixelResult):
        if result is self._last_result:
            return
        self._last_result = result
        self._progress_active = False
        self.progressBarFinished()
        self._worker = None
        if result.generation != self._generation:
            if self._restart_pending:
                self._restart_pending = False
                self.commit.now()
            return
        self._send_result(result)

    def _on_worker_failed(self, tb: str, generation: int):
        self._progress_active = False
        self.progressBarFinished()
        self._worker = None
        if generation != self._generation:
            if self._restart_pending:
                self._restart_pending = False
                self.commit.now()
            return
        self.Outputs.features.send(None)
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        self.Error.extraction_failed(last)

    def _send_result(self, result: _PixelResult):
        from nn_kernel.pixels import pixel_feature_names

        self.Warning.rows_skipped(
            result.n_unreadable + result.skipped_no_path,
            result.n_unreadable,
            result.skipped_no_path,
            shown=bool(result.n_unreadable or result.skipped_no_path),
        )
        if not result.values.size:
            self.Warning.no_images()
            self.Outputs.features.send(None)
            return

        names = pixel_feature_names(result.size)
        source = self.data.domain
        keep = np.asarray(result.readable_source, dtype=int)
        domain = Domain(
            [ContinuousVariable(name) for name in names],
            class_vars=list(source.class_vars),
            metas=list(source.metas),
        )
        out = Table.from_numpy(
            domain,
            result.values,
            self.data.Y[keep] if source.class_vars else None,
            self.data.metas[keep],
        )
        self.Outputs.features.send(out)

    def wait_for_extraction(self, timeout_ms: int = 30000):
        """Wait for the current and any pending extraction worker."""
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
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWNNImagePixels).run()
