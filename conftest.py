# Route test collection by environment.
#
# Kernel tests must run in a plain Python without Orange or Qt. Widget tests
# need the Orange interpreter with a working Qt binding (offscreen is enough),
# so they are collected only when Orange and Qt actually import. A module
# spec is not enough: partial installs exist.
import importlib


def _orange_usable() -> bool:
    try:
        import AnyQt.QtWidgets  # noqa: F401
        import Orange  # noqa: F401

        return True
    except Exception:
        return False


if not _orange_usable():
    collect_ignore = ["tests/widgets"]
else:
    collect_ignore = []

if importlib.util.find_spec("cv2") is None:
    collect_ignore += [
        f"tests/test_{name}.py"
        for name in (
            "colour",
            "extract",
            "photometry",
            "preprocess",
            "profiles",
            "spatial",
            "stability",
            "symmetry",
            "texture",
            "topology",
        )
    ]
