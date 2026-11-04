"""Widget names and toolbox categories."""

from orangecanvas.registry import WidgetRegistry
from orangewidget.workflow.discovery import WidgetDiscovery


def test_widget_category_layout():
    discovery = WidgetDiscovery(registry=WidgetRegistry())
    descriptions = {
        desc.name: desc.category
        for desc in discovery.iter_widget_descriptions(
            "nn_suite.widgets", category_name="Language Models"
        )
    }

    assert descriptions == {
        "Image Features": "Image Analytics",
        "Image Pixels": "Image Analytics",
        "LM Text Loader": "Language Models",
        "LM Pre-trainer": "Language Models",
        "LM Generator": "Language Models",
        "LoRA Pre-trainer": "Language Models",
        "LoRA SFT": "Language Models",
        "LM Vocab Viewer": "Language Models",
        "Neural Network Viewer": "Visualize",
    }
