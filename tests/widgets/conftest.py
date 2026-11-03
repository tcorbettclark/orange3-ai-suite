"""Widget test fixtures: offscreen Qt, no display needed (nnviewer precedent).

This directory is only collected when the Orange interpreter runs the suite
(see root conftest.py), so importing Orange is safe here — but the conftest
itself is still IMPORTED when a plain Python runs pytest, so bail cleanly.
"""

import importlib.util

import pytest


def _orange_usable() -> bool:
    try:
        import Orange  # noqa: F401
        import AnyQt.QtWidgets  # noqa: F401

        return True
    except Exception:
        return False


if not _orange_usable():
    pytest.skip("widget tests need the Orange interpreter", allow_module_level=True)

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from AnyQt.QtWidgets import QApplication
from Orange.data import Domain, StringVariable, Table


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


# ---- fixtures for the folded orange3-nnviewer tests (tests/widgets/nn_viewer/) ----

import warnings as _warnings  # noqa: E402

from Orange.data import ContinuousVariable  # noqa: E402
from Orange.regression.neural_network import NNRegressionLearner  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402


@pytest.fixture(scope="session")
def make_model():
    """Train (and cache) an MLP regression model of the given shape."""
    models = {}

    def make(n_features=21, hidden=(3,), seed=0):
        key = (n_features, hidden, seed)
        if key not in models:
            rs = np.random.RandomState(seed)
            dom = Domain(
                [ContinuousVariable(f"F{i:02d}") for i in range(n_features)],
                ContinuousVariable("Target"),
            )
            data = Table.from_numpy(dom, rs.rand(150, n_features), rs.rand(150, 1))
            # The fixture only needs a model with the right layer shapes,
            # not a converged one; keep the test output clean.
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=ConvergenceWarning)
                with np.errstate(all="ignore"):
                    models[key] = NNRegressionLearner(
                        hidden_layer_sizes=hidden, max_iter=50, random_state=seed
                    )(data)
        return models[key]

    return make


class _DummyHoverEvent:
    """Stand-in for QGraphicsSceneHoverEvent, which PyQt6 refuses to
    instantiate. The widget's handlers run their own code first and then
    forward the event via super(); that forward raises TypeError on the
    dummy, which `fire_hover` is allowed to swallow -- and *only* that."""

    def screenPos(self):
        from AnyQt.QtCore import QPoint

        return QPoint(100, 100)


@pytest.fixture()
def fire_hover():
    import traceback as _traceback

    def fire(handler):
        try:
            handler(_DummyHoverEvent())
        except TypeError as exc:
            # PyQt6 raises TypeError when the handler forwards the dummy
            # event to super().hover*Event(). That forward is the only
            # TypeError this fixture may swallow: verify the innermost
            # traceback frame is the handler's `super()` line, so a
            # genuine bug in the handler body is not masked.
            tb = _traceback.extract_tb(exc.__traceback__)
            innermost = tb[-1]
            assert innermost.name.startswith("hover"), tb
            assert innermost.line and "super()" in innermost.line, (
                f"TypeError raised by handler itself, not the super() forward: {exc}"
            )

    return fire


@pytest.fixture()
def widget(qapp, make_model):
    """An OWNeuralNetworkViewer instance (the folded nn-viewer widget)."""
    from orangecontrib.nn_suite.widgets.ow_nn_viewer import OWNeuralNetworkViewer

    w = OWNeuralNetworkViewer()
    w.show()
    yield w
    w.close()


@pytest.fixture()
def qapp_process_events(qapp):
    """Deliver queued cross-thread signals after a worker finishes."""
    def _drain():
        qapp.processEvents()
    return _drain


@pytest.fixture()
def texts():
    return [
        "the cat sat on the mat. the cat saw the dog.",
        "a dog and a cat ran. the dog saw the cat.",
    ]


@pytest.fixture()
def text_table(texts):
    """A plain Table with one string meta column (what Import Documents emits)."""
    domain = Domain([], metas=[StringVariable("text")])
    metas = np.asarray(texts, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.empty((len(texts), 0)), None, metas)


@pytest.fixture()
def capture():
    """Records values sent through an Output signal: `sent = capture(w.Outputs.x)`."""
    def _capture(output):
        sent = []
        original = output.send

        def send(value, *args, **kwargs):
            sent.append(value)
            original(value, *args, **kwargs)

        output.send = send
        return sent

    return _capture