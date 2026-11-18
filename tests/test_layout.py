"""Where image code lives, and which modules are support code.

The kernel is organised by domain; ``images`` is the image domain, and inside it
the two consumers (``features`` and ``pixels``) share only decoding. A leading
underscore marks support code that nothing outside ``images`` may import.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "orangecontrib" / "ai_suite" / "kernel"
IMAGES = KERNEL / "images"

PUBLIC = {"catalogue", "description", "extraction", "feature", "groups", "guide", "options"}
SUPPORT = {"_conventions", "_measurements", "_preprocessing"}


def _feature_modules() -> set[str]:
    files = {p.stem for p in (IMAGES / "features").glob("*.py")}
    directories = {
        d.name for d in (IMAGES / "features").iterdir() if d.is_dir() and not d.name.startswith("_")
    }
    return files | directories


def test_the_image_domain_has_three_parts():
    assert {p.name for p in IMAGES.iterdir() if p.suffix == ".py"} == {
        "__init__.py",
        "decode.py",
        "pixel_values.py",
    }
    assert (IMAGES / "features").is_dir()
    # no second home for the same code
    assert not (KERNEL / "features").exists()
    assert not (IMAGES / "features" / "decode.py").exists()


def test_every_feature_module_is_declared_public_or_support():
    assert _feature_modules() == PUBLIC | SUPPORT | {"__init__"}


def test_support_modules_are_private_and_public_ones_are_not():
    for name in SUPPORT:
        assert (IMAGES / "features" / f"{name}.py").exists()
        assert name.startswith("_")
    for name in PUBLIC - {"catalogue"}:
        assert (IMAGES / "features" / f"{name}.py").exists()
        assert not name.startswith("_")


def test_production_code_outside_images_uses_only_the_public_modules():
    # Tests may reach inside on purpose; a widget or an example may not.
    offenders = []
    sources = [
        *(ROOT / "orangecontrib").rglob("*.py"),
        *(ROOT / "examples").rglob("*.py"),
    ]
    for path in sources:
        if IMAGES in path.parents or "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                if SUPPORT & set(parts):
                    offenders.append(f"{path.relative_to(ROOT)}: {node.module}")
    assert offenders == []
