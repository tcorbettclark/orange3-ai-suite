"""Where kernel code lives, and which modules are support code.

The kernel is organised by domain; ``images`` is the image domain, holding the
image pipeline (``decode``, ``resize``, ``flatten``, and ``pixels`` composing
them) and the ``features`` package built on it. A leading underscore marks
support code that nothing outside ``images`` may import.

``lora`` puts everything that knows MLX behind ``lora/mlx``: the domain package
above it - options, data, the split, the run's scratch, the artifact - imports
nothing from ``mlx`` or ``mlx_lm``, and the boundary package never reaches back
up into its parent.
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


def test_the_image_domain_has_five_parts():
    assert {p.name for p in IMAGES.iterdir() if p.suffix == ".py"} == {
        "__init__.py",
        "decode.py",
        "flatten.py",
        "pixels.py",
        "resize.py",
    }
    assert (IMAGES / "features").is_dir()
    # no second home for the same code
    assert not (KERNEL / "features").exists()
    assert not (IMAGES / "features" / "decode.py").exists()


def test_the_kernel_root_holds_domains_and_their_shared_steps_only():
    # A domain package per subject, plus the shared steps that two domains need:
    # the table is the decision, and a new file here has to be argued for.
    assert {p.name for p in KERNEL.iterdir() if p.is_dir() and not p.name.startswith("__")} == {
        "assets",
        "images",
        "lm",
        "lora",
        "text",
    }
    assert {p.name for p in KERNEL.glob("*.py")} == {"__init__.py", "_holdout.py"}


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


# ---- the MLX boundary ------------------------------------------------------

LORA = KERNEL / "lora"
LORA_MLX = LORA / "mlx"


def _imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """(module, relative level) for every import in a file; level 0 is absolute."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


def test_only_the_mlx_boundary_names_mlx():
    # mlx is Apple-Silicon-only, so a stray import above the boundary would make
    # the whole kernel unimportable elsewhere - and would hide the coupling from
    # the one module that is supposed to own it.
    offenders = []
    for path in (ROOT / "orangecontrib").rglob("*.py"):
        if LORA_MLX in path.parents or "__pycache__" in path.parts:
            continue
        for module, level in _imports(path):
            # relative imports name our own subpackage; only the library itself
            # (an absolute import) is forbidden out here
            if level == 0 and (module == "mlx" or module.startswith(("mlx.", "mlx_lm"))):
                offenders.append(f"{path.relative_to(ROOT)}: {module}")
    assert offenders == []


def test_the_mlx_boundary_never_reaches_back_up_into_lora():
    # A leaf importing its parent is an arrow pointing the wrong way: the domain
    # package may know the boundary, never the reverse.
    offenders = []
    for path in LORA_MLX.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for module, level in _imports(path):
            if level > 1:
                offenders.append(f"{path.relative_to(ROOT)}: {'.' * level}{module}")
            elif module.startswith("orangecontrib.ai_suite.kernel.lora.") and not module.startswith(
                "orangecontrib.ai_suite.kernel.lora.mlx"
            ):
                offenders.append(f"{path.relative_to(ROOT)}: {module}")
    assert offenders == []


def test_the_mlx_boundary_is_a_package_with_a_facade():
    # What the domain may use is the subpackage's __all__, so the boundary has a
    # contract rather than a directory of internals.
    from orangecontrib.ai_suite.kernel.lora import mlx

    assert {"Run", "TrainingSpec", "Event", "load_model", "stream_text"} <= set(mlx.__all__)
    for name in mlx.__all__:
        assert hasattr(mlx, name), name
    assert {p.name for p in LORA_MLX.glob("*.py")} == {
        "__init__.py",
        "child.py",
        "layout.py",
        "protocol.py",
        "runtime.py",
        "train.py",
    }
