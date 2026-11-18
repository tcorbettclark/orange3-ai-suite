"""BritLLM Models: download and cache BritLLM base models for the LoRA widgets.

One pinned 4-bit MLX BritLLM. Click Download to fetch it into the Hugging Face
cache and verify its weights; the local directory is sent on the Model path
output. The LoRA widgets receive that path on their Base model input, so the
handoff is a signal rather than a lookup in a shared cache.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path

from AnyQt.QtCore import Qt, QThread, QUrl, pyqtSignal
from AnyQt.QtGui import QDesktopServices
from AnyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
)
from Orange.widgets import gui
from Orange.widgets.widget import Msg, Output, OWWidget

from orangecontrib.ai_suite.kernel.assets import (
    MODEL_ASSETS,
    Asset,
    Progress,
    cached_path,
    ensure,
    hub_cache,
    verify,
)

_NAMES = tuple(MODEL_ASSETS)


class _Worker(QThread):
    """Run one blocking asset task; progress and outcome travel as signals."""

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task: Callable[[Progress], object], parent=None):
        super().__init__(parent)
        self.task = task

    def run(self) -> None:
        try:
            self.finished_ok.emit(self.task(self._report))
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _report(self, done: int, total: int) -> None:
        self.progress.emit(int(done), int(total))


class OWBritLLMModels(OWWidget):
    """Download the 4-bit BritLLM model and pass its local path downstream.

    The model comes from one pinned Hugging Face repository. The cache is the
    standard Hugging Face hub cache (``HF_HUB_CACHE``, else ``$HF_HOME/hub``),
    so another HF consumer can reuse it and a lab can share it by setting
    ``HF_HOME``. The output is the snapshot directory, not a name the LoRA
    widgets would have to resolve themselves.
    """

    name = "BritLLM Models"
    description = "Download and cache the BritLLM base models used by the LoRA widgets."
    icon = "icons/britllm_models.svg"
    priority = 14
    keywords = [  # noqa: RUF012
        "britllm",
        "model",
        "download",
        "cache",
        "lora",
        "mlx",
    ]

    class Outputs:
        model = Output("Model path", object, auto_summary=False)

    class Warning(OWWidget.Warning):
        not_downloaded = Msg("The model is not downloaded yet; press Download.")

    class Error(OWWidget.Error):
        download_failed = Msg("Download failed:\n{}")
        hub_missing = Msg(
            "Downloading needs huggingface_hub: install orange3-ai-suite[lora]."
        )

    def __init__(self):
        super().__init__()
        self.worker: _Worker | None = None
        self.path: str | None = None

        box = gui.widgetBox(self.controlArea, "Model")
        self.model_combo = QComboBox()
        for name in _NAMES:
            asset = MODEL_ASSETS[name]
            self.model_combo.addItem(
                f"BritLLM 3B v0.1 - 4-bit MLX ({asset.size_bytes / 1e9:.1f} GB)", name
            )
        self.model_combo.currentIndexChanged.connect(lambda _: self._refresh())
        box.layout().addWidget(self.model_combo)

        self.help_label = QLabel(
            "Start with this 4-bit model. Other MLX models (8-bit or unquantised) "
            "work if the machine has the memory."
        )
        self.help_label.setWordWrap(True)
        box.layout().addWidget(self.help_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        box.layout().addWidget(self.status_label)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.layout().addWidget(self.path_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        box.layout().addWidget(self.progress)

        buttons = QHBoxLayout()
        self.download_button = QPushButton()
        self.download_button.clicked.connect(self._download)
        self.verify_button = QPushButton("Verify")
        self.verify_button.clicked.connect(self._verify)
        self.open_button = QPushButton("Open cache")
        self.open_button.clicked.connect(self._open_cache)
        for button in (self.download_button, self.verify_button, self.open_button):
            buttons.addWidget(button)
        box.layout().addLayout(buttons)

        cache_label = QLabel(f"Cache: {hub_cache()}")
        cache_label.setWordWrap(True)
        cache_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.layout().addWidget(cache_label)

        self.mainArea.layout().addWidget(
            QLabel(
                "The snapshot is passed on the Model path output. Connect it to "
                "LoRA Pre-trainer or LoRA SFT's Base model input; their Base model "
                "field can also take any local MLX model directory or HF id."
            )
        )
        self.resize(520, 300)
        self._refresh()

    # ---- state -------------------------------------------------------------

    def _asset(self) -> Asset:
        return MODEL_ASSETS[self.model_combo.currentData()]

    def _refresh(self) -> None:
        asset = self._asset()
        self.Warning.not_downloaded.clear()
        path = cached_path(asset)
        self.path = str(path) if path is not None else None
        self.verify_button.setEnabled(path is not None)
        if path is None:
            self.status_label.setText("Not downloaded")
            self.path_label.setText("")
            self.download_button.setText(f"Download {asset.size_bytes / 1e9:.1f} GB")
            self.Warning.not_downloaded()
        else:
            self.status_label.setText(f"Cached and verified candidate for {asset.name}")
            self.path_label.setText(str(path))
            self.download_button.setText("Use cached model")
        self._emit()

    def _emit(self) -> None:
        if self.path is not None:
            self.Outputs.model.send(self.path)

    # ---- actions -----------------------------------------------------------

    def _download(self) -> None:
        if self.path is not None:
            self._emit()
            return
        asset = self._asset()
        self._start(
            lambda report: ensure(asset, on_progress=report),
            f"Downloading {asset.name}...",
            total=asset.size_bytes,
        )

    def _verify(self) -> None:
        if self.path is None:
            return
        asset = self._asset()
        path = self.path
        self._start(lambda report: verify(asset, path), "Verifying weights...", total=None)

    def _start(
        self,
        task: Callable[[Progress], object],
        status: str,
        total: int | None,
    ) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.Error.download_failed.clear()
        self.Error.hub_missing.clear()
        self.Warning.not_downloaded.clear()
        self.status_label.setText(status)
        self.download_button.setEnabled(False)
        self.verify_button.setEnabled(False)
        self.progress.setVisible(True)
        if total is None:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(0)

        worker = _Worker(task)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._worker_finished)
        self.worker = worker
        worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
            self.status_label.setText(
                f"Downloading {done / 1e9:.2f} of {total / 1e9:.2f} GB"
                f" ({done * 100 // total}%)"
            )

    def _on_done(self, result) -> None:
        path = Path(str(result))
        self.path = str(path)
        self.progress.setVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.status_label.setText("Cached and verified")
        self.path_label.setText(str(path))
        self.download_button.setText("Use cached model")
        self._emit()
        self.download_button.setEnabled(True)
        self.verify_button.setEnabled(True)

    def _on_failed(self, tb: str) -> None:
        self.progress.setVisible(False)
        clean = tb.strip()
        last = clean.splitlines()[-1] if clean else "unknown error"
        if "huggingface_hub" in last:
            self.Error.hub_missing()
        else:
            self.Error.download_failed(last)
        self.status_label.setText("Failed")
        self.download_button.setEnabled(True)
        self.verify_button.setEnabled(self.path is not None)

    def _worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()

    def _open_cache(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(hub_cache())))

    def onDeleteWidget(self):
        # The download cannot be cancelled; keep the thread alive under the
        # application and let it delete itself when it finishes.
        if self.worker is not None and self.worker.isRunning():
            self.worker.setParent(QApplication.instance())
            self.worker.finished.connect(self.worker.deleteLater)
        super().onDeleteWidget()


if __name__ == "__main__":  # pragma: no cover - dev preview
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWBritLLMModels).run()
