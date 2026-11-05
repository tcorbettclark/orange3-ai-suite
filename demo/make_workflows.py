"""Generate the .ows demo workflows through the canvas Scheme API.

Builds (idempotently):

    demo/mnist_features.ows   (Import Images -> Image Features ->
                               Test & Score + Neural Network, plus
                               Data Table / Column Statistics / Confusion Matrix)
    demo/cloud_features.ows   (same shape, cloud corpus settings)

The canvas API (orangecanvas.scheme) writes the .ows XML we would otherwise
hand-craft; widget settings are pre-seeded per node (Import Images path,
Image Features' working size). On load, the canvas resolves
widgets by qualified_name and matches links by channel name, so the
descriptions here only need faithful channel names/ids — extracted
programmatically from the real widget classes.

    uv run python demo/make_workflows.py
"""

from __future__ import annotations

from pathlib import Path

from orangecanvas.registry.description import (
    InputSignal,
    OutputSignal,
    WidgetDescription,
)
from orangecanvas.scheme import Scheme, SchemeLink, SchemeNode

BASE = Path(__file__).resolve().parent


def make_descriptions() -> dict[str, WidgetDescription]:
    """Widget descriptions with channel names/ids taken from the real classes."""
    from Orange.widgets.data.owfeaturestatistics import OWFeatureStatistics
    from Orange.widgets.data.owtable import OWTable
    from Orange.widgets.evaluate.owconfusionmatrix import OWConfusionMatrix
    from Orange.widgets.evaluate.owtestandscore import OWTestAndScore
    from Orange.widgets.model.owneuralnetwork import OWNNLearner

    from orangecontrib.ai_suite.widgets.ow_image_features import OWImageFeatures
    from orangecontrib.imageanalytics.widgets.owimageimport import OWImportImages

    classes = {
        "Import Images": OWImportImages,
        "Image Features": OWImageFeatures,
        "Neural Network": OWNNLearner,
        "Test & Score": OWTestAndScore,
        "Data Table": OWTable,
        "Column Statistics": OWFeatureStatistics,
        "Confusion Matrix": OWConfusionMatrix,
    }
    descriptions = {}
    for label in (
        "Import Images",
        "Data Table",
        "Column Statistics",
        "Confusion Matrix",
        "Test & Score",
        "Neural Network",
        "Image Features",
    ):
        cls = classes[label]
        inputs = [
            InputSignal(i.name, _type_str(i.type), None, 18, i.id)
            for i in vars(cls.Inputs).values()
            if type(i).__name__ in ("Input", "MultiInput")
        ]
        outputs = [
            OutputSignal(o.name, _type_str(o.type), 16, o.id)
            for o in vars(cls.Outputs).values()
            if type(o).__name__ in ("Output",)
        ]
        descriptions[label] = WidgetDescription(
            name=label,
            category="Image Analytics",
            qualified_name=f"{cls.__module__}.{cls.__name__}",
            id=f"{cls.__module__}",
            inputs=inputs,
            outputs=outputs,
            priority=getattr(cls, "priority", 100),
            icon=getattr(cls, "icon", "icons/default.svg"),
        )
    return descriptions


def _type_str(type_):
    return f"{getattr(type_, '__module__', '')}.{getattr(type_, '__qualname__', type_)}"


def image_import_path_properties(directory: Path):
    """Import Images opens folders via `recent_paths` (RecentPath objects)."""
    from orangewidget.utils.filedialogs import RecentPath

    recent = RecentPath(str(directory), str(directory.name), None)  # type: ignore
    return {"recent_paths": [recent]}


FEATURE_PROPS = "Image Features"


def build_workflow(
    title: str, description: str, image_dir: Path, feature_options: dict, dest: Path
) -> None:
    d = make_descriptions()

    scheme = Scheme(title=title, description=description)
    columns = ["Import Images", "Image Features", "Test & Score", "Confusion Matrix"]

    nodes = {}
    x = 60
    for i, label in enumerate(columns):
        node = SchemeNode(
            d[label],
            title=label,
            position=(x + i * 240, 40),
            properties={},
        )
        scheme.add_node(node)
        nodes[label] = node

    # viewers + learner row below the main flow
    nodes["Data Table"] = SchemeNode(
        d["Data Table"], title="Data Table", position=(780, 260), properties={}
    )
    nodes["Column Statistics"] = SchemeNode(
        d["Column Statistics"], title="Column Statistics", position=(540, 260), properties={}
    )
    nodes["Neural Network"] = SchemeNode(
        d["Neural Network"], title="Neural Network", position=(300, 260), properties={}
    )
    for label in ("Data Table", "Column Statistics", "Neural Network"):
        scheme.add_node(nodes[label])

    nodes["Import Images"].properties = dict(image_import_path_properties(image_dir))
    nodes[FEATURE_PROPS].properties = dict(feature_options)

    def link(source: str, out: str, sink: str, inp: str) -> None:
        source_channel = next(c for c in nodes[source].description.outputs if c.name == out)
        sink_channel = next(c for c in nodes[sink].description.inputs if c.name == inp)
        scheme.add_link(SchemeLink(nodes[source], source_channel, nodes[sink], sink_channel))

    link("Import Images", "Data", FEATURE_PROPS, "Images")
    link(FEATURE_PROPS, "Features", "Test & Score", "Data")
    link("Neural Network", "Learner", "Test & Score", "Learner")
    link(FEATURE_PROPS, "Features", "Data Table", "Data")
    link(FEATURE_PROPS, "Features", "Column Statistics", "Data")
    link("Test & Score", "Evaluation Results", "Confusion Matrix", "Evaluation Results")

    scheme.save_to(str(dest), pickle_fallback=True)  # recent_paths is not literal-serialisable
    print(f"wrote {dest}")


def main() -> None:
    build_workflow(
        "MNIST digits — interpretable features",
        "Import Images (demo/mnist/train) -> Image Features (working "
        "size 56) -> Test & Score with the Neural Network; Data Table, Column "
        "Statistics and Confusion Matrix for inspection.",
        BASE.parent / "demo" / "mnist" / "train",
        {"working_size": 56, "groups_enabled": "ABCDEF"},
        BASE / "mnist_features.ows",
    )
    build_workflow(
        "WMO Cloud Atlas genera — interpretable features",
        "Import Images (data/clouds, fetched by demo/fetch_clouds.py) -> "
        "Image Features -> Test & Score with the Neural Network; inspect "
        "features in Data Table / Column Statistics; Confusion Matrix shows "
        "which genera confound.",
        BASE.parent / "data" / "clouds",
        {"working_size": 128, "groups_enabled": "ABCDEF"},
        BASE / "cloud_features.ows",
    )


if __name__ == "__main__":
    main()
