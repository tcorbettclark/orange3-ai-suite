"""Image Feature Extractor: interpretable features for image classification.

A thin shell over nn_kernel.features (groups A-F, 36 features; spec
image-feature-extractor-widget.md §3-4). The widget adds: options
(preprocessing parameters, feature-group toggles for ablation experiments),
a preview showing the selected image's luminance with its row/column
projection profiles ("makes the feature visible, not just a number"), and
the Orange signal adapters.

Output: one row per image, one column per feature, the input's class
columns inherited, and the input metas (including the image path) kept so
any suspicious row traces back to its source image — debuggability IS the
point of the exercise.

Extraction runs in the GUI thread with a progress bar (feature extraction at
the working size is milliseconds per image — teaching scale); training-scale
background threads are the LM Trainer's pattern.
"""

from __future__ import annotations

import importlib.util

import numpy as np
from Orange.data import ContinuousVariable, Domain, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, OWWidget, Output
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage, QPixmap
from AnyQt.QtWidgets import QCheckBox, QHBoxLayout, QLabel

from nn_suite.adapters import image_paths

GROUP_LABELS = (
    ("A", "A: global photometry"),
    ("B", "B: profile summaries"),
    ("C", "C: spatial mass"),
    ("D", "D: texture"),
    ("E", "E: topology"),
    ("F", "F: symmetry"),
    ("G", "G: colour — the shortcut trap (default off, spec §2)"),
)


class OWImageFeatureExtractor(OWWidget):
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
        no_image_column = Msg(
            "No image column found — connect the Images output of Import Images."
        )
        rows_skipped = Msg("{} row(s) skipped ({} unreadable, {} without a path).")
        all_groups_off = Msg("All feature groups are switched off.")

    class Error(OWWidget.Error):
        imageanalytics_missing = Msg(
            "Orange3-ImageAnalytics is required (Options > Add-ons)."
        )
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
    scale_features = Setting(False)
    preview_index = Setting(0)

    def __init__(self):
        super().__init__()
        self.data = None
        self._paths: list[str] = []

        prep_box = gui.widgetBox(self.controlArea, "Preprocessing")
        gui.spin(prep_box, self, "working_size", 32, 512, label="Working size (square)")
        gui.checkBox(prep_box, self, "centre_crop", "Centre-crop to square")
        gui.checkBox(prep_box, self, "contrast_normalise", "Contrast-normalise")
        gui.checkBox(prep_box, self, "denoise", "Denoise (median filter)")
        gui.spin(prep_box, self, "denoise_ksize", 3, 5, label="Denoise strength")

        param_box = gui.widgetBox(self.controlArea, "Feature parameters")
        gui.doubleSpin(param_box, self, "peak_prominence_frac", 0.01, 0.5, 0.01,
                       label="Peak prominence (of range)")
        gui.doubleSpin(param_box, self, "edge_density_frac", 0.1, 1.0, 0.05,
                       label="Edge density (of gradient p99)")

        group_box = gui.widgetBox(self.controlArea, "Feature groups (ablation)")
        self._group_checks: dict[str, QCheckBox] = {}
        for letter, label in GROUP_LABELS:
            check = QCheckBox(label, group_box)
            check.setChecked(letter in self.groups_enabled)
            check.toggled.connect(self._sync_group_settings)
            group_box.layout().addWidget(check)
            self._group_checks[letter] = check

        gui.auto_commit(self.controlArea, self, "auto_apply", "Extract features",
                        commit=self.commit)
        gui.checkBox(self.controlArea, self, "scale_features",
                     "Scale features (z-score, whole-batch; measured +0.03 CA on "
                     "the cloud demo — note: stats come from the whole table, "
                     "so keep Test & Score's folds in mind)")

        self._setup_preview()
        self.resize(950, 700)

    def _sync_group_settings(self):
        self.groups_enabled = "".join(
            letter for letter, _ in GROUP_LABELS if self._group_checks[letter].isChecked()
        )

    def _selected_groups(self) -> str:
        return self.groups_enabled

    # ---- preview -----------------------------------------------------------

    def _setup_preview(self):
        import pyqtgraph as pg

        container = gui.widgetBox(self.mainArea, "Preview (luminance + projections)")
        top = QHBoxLayout()
        container.layout().addLayout(top)
        self._preview_label = QLabel("No images.")
        self._preview_label.setFixedSize(200, 200)
        self._preview_label.setAlignment(Qt.AlignCenter)
        top.addWidget(self._preview_label)
        self._preview_spin = gui.spin(container, self, "preview_index", 0, 0,
                                      label="Image #", callback=self._update_preview)
        self._profiles = pg.PlotWidget(background="w")
        self._profiles.addLegend(offset=(10, 10))
        self._profiles.setLabel("bottom", "normalised position across the image")
        self._profiles.setLabel("left", "mean luminance (row ->, column ^)")
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
            lum = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                             self._feature_options())
        except ImportError:
            return  # preview is a nicety; extraction errors carry the install hint
        h, w = lum.shape
        image = QImage(lum.tobytes(), w, h, w, QImage.Format_Grayscale8)
        self._preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                196, 196, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        row_profile = lum.mean(axis=1)
        col_profile = lum.mean(axis=0)
        self._row_curve.setData(np.linspace(0, 1, len(col_profile)), row_profile)
        self._col_curve.setData(np.linspace(0, 1, len(col_profile)), col_profile)

    @Inputs.images
    def set_data(self, table):
        self.data = table
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
            raise ValueError("OpenCV not available — install the [image] extra")
        groups = self._selected_groups()
        if not groups:
            self.Warning.all_groups_off()
            self.Outputs.features.send(None)
            return
        try:
            _, paths = image_paths(self.data)
        except ImportError as exc:
            raise ValueError(f"Orange3-ImageAnalytics required ({exc.name})") from exc

        kept = [(i, p) for i, p in enumerate(paths) if p]
        skipped_no_path = len(paths) - len(kept)

        import cv2
        from nn_kernel.features import extract_features

        rows, readable = [], []
        n_unreadable = 0
        self.progressBarInit()
        try:
            for n, (i, path) in enumerate(kept):
                self.progressBarSet(100.0 * n / max(len(kept), 1))
                bgr = cv2.imread(path)
                if bgr is None:
                    n_unreadable += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rows.append(extract_features(rgb, self._feature_options(), groups=groups))
                readable.append(i)
        finally:
            self.progressBarFinished()

        self.Warning.rows_skipped(
            n_unreadable + skipped_no_path, n_unreadable, skipped_no_path,
            shown=bool(n_unreadable or skipped_no_path),
        )
        self._paths = [path for _, path in kept]
        self._clamp_preview()
        if not rows:
            self.Outputs.features.send(None)
            self._update_preview()
            return

        ordered = [
            name
            for letter, _ in GROUP_LABELS
            if letter in groups
            for name in self._registry_group_names(letter)
        ]
        values = np.asarray(
            [[vec[name] for name in ordered] for vec in rows], dtype=np.float64
        )
        if self.scale_features and len(values) > 1:
            stds = values.std(axis=0)
            values = (values - values.mean(axis=0)) / np.where(stds < 1e-12, 1.0, stds)
        source = self.data.domain
        kept_idx = np.asarray(readable, dtype=int)
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

    @staticmethod
    def _registry_group_names(letter):
        from nn_kernel.features import group_names

        return group_names(letter)

    def _clamp_preview(self):
        n = len(self._paths)
        self.preview_index = min(self.preview_index, max(n - 1, 0))
        self._preview_spin.setMaximum(max(n - 1, 0))