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