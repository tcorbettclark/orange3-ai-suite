"""BritLLM Models widget: an explicit click, and an explicit path output."""

import pytest

from orangecontrib.ai_suite.widgets import ow_britllm_models as mod
from orangecontrib.ai_suite.widgets.ow_britllm_models import OWBritLLMModels


def test_nothing_is_downloaded_before_the_click(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "cached_path", lambda asset: None)
    monkeypatch.setattr(mod, "ensure", lambda *args, **kwargs: calls.append(1))
    w = OWBritLLMModels()
    w.show()
    assert w.path is None
    assert w.Warning.not_downloaded.is_shown()
    assert calls == []


def test_download_sends_the_local_path(
    qapp, tmp_path, monkeypatch, capture, qapp_process_events
):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(mod, "cached_path", lambda asset: None)

    def fake_ensure(asset, on_progress=None):
        on_progress(5, 10)
        on_progress(10, 10)
        return model

    monkeypatch.setattr(mod, "ensure", fake_ensure)
    w = OWBritLLMModels()
    w.show()
    sent = capture(w.Outputs.model)
    w.download_button.click()
    worker = w.worker
    assert worker is not None
    worker.wait(2000)
    qapp_process_events()
    assert w.path == str(model)
    assert sent[-1] == str(model)
    assert w.status_label.text() == "Cached and verified"
    assert not w.Error.download_failed.is_shown()


def test_a_cached_model_is_used_on_click(
    qapp, tmp_path, monkeypatch, capture, qapp_process_events
):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(mod, "cached_path", lambda asset: model)
    monkeypatch.setattr(mod, "ensure", lambda *args, **kwargs: pytest.fail("must not download"))
    w = OWBritLLMModels()
    w.show()
    sent = capture(w.Outputs.model)
    assert w.download_button.text() == "Use cached model"
    w.download_button.click()
    qapp_process_events()
    assert sent[-1] == str(model)
