# Kernel-only vs widget tests:
# - kernel tests must run in a plain Python (no Orange installed) — REPO-CONTEXT §4
# - widget tests run with the Orange interpreter (offscreen Qt)
# Route collection by environment. "Orange usable" means Orange AND a Qt
# binding actually import — find_spec() alone is not enough (partial installs).
import importlib


def _orange_usable() -> bool:
    try:
        import Orange  # noqa: F401
        import AnyQt.QtWidgets  # noqa: F401

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
            "colour", "extract", "photometry", "preprocess", "profiles", "spatial",
            "stability", "symmetry", "texture", "topology",
        )
    ]