"""Where kernel code lives, and which modules are support code.

The kernel is organised by domain; ``images`` is the image domain, holding the
image pipeline (``decode``, ``resize``, ``flatten``) and the ``features`` package
built on it. A leading underscore marks support code that nothing outside
``images`` may import.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "orangecontrib" / "ai_suite" / "kernel"
IMAGES = KERNEL / "images"

TEXT = KERNEL / "text"

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
        "flatten.py",
        "resize.py",
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


def test_the_text_package_is_a_facade_over_private_steps():
    assert {p.name for p in TEXT.glob("*.py")} == {
        "__init__.py",
        "_clean.py",
        "_qa.py",
        "_split.py",
    }
    assert not (KERNEL / "text.py").exists()


def test_the_text_facade_exports_every_public_name_and_nothing_private():
    from orangecontrib.ai_suite.kernel import text as facade

    assert facade.__all__ == list(dict.fromkeys(facade.__all__))  # no duplicates
    for name in facade.__all__:
        assert hasattr(facade, name), name
    assert not [name for name in facade.__all__ if name.startswith("_")]
