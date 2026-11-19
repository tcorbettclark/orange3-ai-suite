"""Where kernel code lives, and which modules are support code.

``images`` holds the image pipeline (``decode``, ``resize``, ``flatten`` and ``pixels``
composing them) and the ``features`` package built on it, and a leading underscore
marks support code that nothing outside ``images`` may import. ``lora`` puts everything
that knows MLX behind ``lora/mlx``, which never reaches back up into its parent.
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

#: The boundary is the modules whose names begin with this: mlx-lm's files, its child
#: process, its wire format and its in-process runtime, in the same package as the
#: domain so that one predicate answers "is this the boundary?".
BOUNDARY_PREFIX = "mlx_"


def _imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """The ``(module, relative level)`` pairs of every import in a file; level 0 is absolute."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


def _boundary() -> set[pathlib.Path]:
    return {p for p in LORA.glob("*.py") if p.stem.startswith(BOUNDARY_PREFIX)}


def test_only_the_boundary_modules_name_mlx():
    # mlx is Apple-Silicon-only, so a stray import above the boundary would make
    # the whole kernel unimportable elsewhere - and would hide the coupling from
    # the modules that are supposed to own it.
    boundary = _boundary()
    offenders = []
    for path in (ROOT / "orangecontrib").rglob("*.py"):
        if path in boundary or "__pycache__" in path.parts:
            continue
        for module, level in _imports(path):
            # relative imports name our own package; only the library itself
            # (an absolute import) is forbidden out here
            if level == 0 and (module == "mlx" or module.startswith(("mlx.", "mlx_lm"))):
                offenders.append(f"{path.relative_to(ROOT)}: {module}")
    assert offenders == []


def test_the_boundary_modules_never_import_the_rest_of_lora():
    # A boundary module may import its siblings and nothing else: the domain may know
    # the boundary, never the reverse.
    boundary = _boundary()
    siblings = {path.stem for path in boundary}
    offenders = []
    for path in boundary:
        for module, level in _imports(path):
            # `from .mlx_protocol import ...` is a sibling; `from .data import ...` is not
            if level == 1 and module and module not in siblings:
                offenders.append(f"{path.name}: .{module}")
            # a deeper relative hop leaves the package; an absolute one names it
            elif level > 1:
                offenders.append(f"{path.name}: {'.' * level}{module}")
            elif module.startswith("orangecontrib.ai_suite.kernel.lora."):
                tail = module.rsplit(".", 1)[-1]
                if tail not in siblings:
                    offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_the_child_is_a_program_and_nothing_imports_it():
    # `mlx_child` runs as ``python -m ...lora.mlx_child``; the parent starts it by
    # name and must not import it, or the child's imports become the package's.
    import importlib.util

    from orangecontrib.ai_suite.kernel.lora import mlx_process

    found = importlib.util.find_spec(mlx_process._CHILD)
    assert found is not None and found.origin.endswith("/mlx_child.py")  # the name resolves

    offenders = []
    for path in (ROOT / "orangecontrib").rglob("*.py"):
        if path.name == "mlx_child.py" or "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            else:
                continue
            if any(name.rsplit(".", 1)[-1] == "mlx_child" for name in names):
                offenders.append(f"{path.relative_to(ROOT)}: {ast.unparse(node)}")
    assert offenders == []


def test_the_boundary_is_these_modules():
    # The list is the decision: a new boundary module is a new row, and a module that
    # stops naming mlx-lm belongs on the other side of the prefix.
    assert {path.stem for path in _boundary()} == {
        "mlx_child",
        "mlx_layout",
        "mlx_process",
        "mlx_protocol",
        "mlx_runtime",
        "mlx_spec",
    }
    assert {
        path.stem for path in LORA.glob("*.py") if not path.stem.startswith(BOUNDARY_PREFIX)
    } == {"__init__", "artifact", "data", "model", "run"}
