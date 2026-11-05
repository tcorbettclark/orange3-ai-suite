"""Widget test fixtures: offscreen Qt, no display needed.

This directory is only collected when the Orange interpreter runs the suite
(see root conftest.py), so importing Orange is safe here — but the conftest
itself is still IMPORTED when a plain Python runs pytest, so bail cleanly.
"""

import pytest


def _orange_usable() -> bool:
    try:
        import AnyQt.QtWidgets  # noqa: F401
        import Orange  # noqa: F401

        return True
    except Exception:
        return False


if not _orange_usable():
    pytest.skip("widget tests need the Orange interpreter", allow_module_level=True)

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The offscreen platform's font database default family is literally
# "Sans Serif"; its one-time alias-population message goes straight to stderr
# (not a Python warning, so pytest filterwarnings can't catch it) — real
# canvas runs (cocoa) never show it. Silence it in the test harness only.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts*=false")

import numpy as np  # noqa: E402
from AnyQt.QtWidgets import QApplication  # noqa: E402
from Orange.data import DiscreteVariable, Domain, StringVariable, Table  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


# ---- fixtures for the neural network viewer tests (tests/widgets/neural_network_viewer/) ----

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
    """An OWNeuralNetworkViewer instance."""
    from orangecontrib.ai_suite.widgets.ow_neural_network_viewer import OWNeuralNetworkViewer

    w = OWNeuralNetworkViewer()
    w.show()
    yield w
    w.close()


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    """flat / noisy / disc images on disk, PNG, 64x64."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("images")
    paths = []

    def save(arr, name):
        from PIL import Image

        path = d / name
        Image.fromarray(arr.astype(np.uint8)).save(path)
        paths.append(path)

    # flat: solid mid-gray (contrast-normalises to all-zeros)
    save(np.full((64, 64, 3), 128), "flat1.png")
    save(np.full((64, 64, 3), 130), "flat2.png")
    # noisy: strong texture
    save(rng.uniform(0, 255, size=(64, 64, 3)), "noisy1.png")
    save(rng.uniform(0, 255, size=(64, 64, 3)), "noisy2.png")
    # disc: a bright disc on black (topology-friendly)
    yy, xx = np.mgrid[0:64, 0:64]
    disc = np.zeros((64, 64, 3))
    disc[(yy - 32) ** 2 + (xx - 32) ** 2 <= 20**2] = 255
    save(disc, "disc.png")
    return d, paths


@pytest.fixture()
def images_table(image_dir):
    """Import-Images-style table: path meta marked type='image' with origin."""
    d, paths = image_dir
    var = StringVariable("image")
    var.attributes["type"] = "image"
    var.attributes["origin"] = str(d)
    domain = Domain(
        [],
        class_vars=[DiscreteVariable("class", values=("flat", "noisy", "disc"))],
        metas=[var],
    )
    names = [p.name for p in paths]
    classes = [0, 0, 1, 1, 2]
    metas = np.asarray(names, dtype=object).reshape(-1, 1)
    return Table.from_numpy(domain, np.empty((len(names), 0)), np.asarray(classes), metas)


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
