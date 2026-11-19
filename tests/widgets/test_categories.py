"""Widget names and toolbox categories."""

from orangecanvas.registry import WidgetRegistry
from orangewidget.workflow.discovery import WidgetDiscovery


def test_widget_category_layout():
    discovery = WidgetDiscovery(registry=WidgetRegistry())
    descriptions = {
        desc.name: desc.category
        for desc in discovery.iter_widget_descriptions(
            "orangecontrib.ai_suite.widgets", category_name="Language Models"
        )
    }

    assert descriptions == {
        "BritLLM Models": "Language Models",
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


def test_the_readme_widget_table_lists_exactly_the_registered_widgets():
    """The README widget table and the registered widgets are the same list."""
    import pathlib

    readme = (pathlib.Path(__file__).resolve().parents[2] / "README.md").read_text()
    table = readme.split("## Widgets", 1)[1].split("\nEach widget class", 1)[0]
    rows = {
        row.split("|")[1].strip()
        for row in table.splitlines()
        if row.startswith("|") and not row.startswith(("| Widget", "|---"))
    }

    discovery = WidgetDiscovery(registry=WidgetRegistry())
    registered = {
        desc.name
        for desc in discovery.iter_widget_descriptions(
            "orangecontrib.ai_suite.widgets", category_name="Language Models"
        )
    }
    assert rows == registered
