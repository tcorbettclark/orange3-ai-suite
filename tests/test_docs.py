"""The test that every name the prose puts in backticks exists in the source tree.

The prose is the README, the load/save note and the tutorials; the installed
dependency is read too. The sweep takes each lower-case ``snake_case`` token in
backticks and looks for it in ``orangecontrib``, ``demo``, ``tests`` and the
``edu_ai_suite`` package.
"""

import ast
import pathlib
import re

import edu_ai_suite

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIRS = [
    ROOT / "orangecontrib",
    ROOT / "demo",
    ROOT / "tests",
    pathlib.Path(edu_ai_suite.__file__).parent,
]
PROSE = [ROOT / "README.md", ROOT / "LOAD_SAVE_MODELS.md"] + sorted(
    (ROOT / "tutorials").glob("*.md")
)

#: A lower-case name with at least one underscore: what the code, not prose, calls
#: a thing. Bare words, CamelCase class names and upper-case constants are out of
#: scope - this test is about the names a reader would grep for.
_CODE_TOKEN = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")


def _source_identifiers() -> set[str]:
    names: set[str] = set()
    for directory in SOURCE_DIRS:
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # pragma: no cover - not this test's job
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.keyword) and node.arg:
                    names.add(node.arg)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # names that travel as strings: a feature name, a data key
                    names.add(node.value)
    return names


def _prose_tokens() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in PROSE:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for token in _CODE_TOKEN.findall(line):
                found.setdefault(token, []).append(f"{path.name}:{number}")
    return found


def test_every_code_name_the_prose_uses_exists_in_the_source():
    identifiers = _source_identifiers()
    invented = {
        token: where for token, where in _prose_tokens().items() if token not in identifiers
    }
    assert invented == {}, "the prose names things the code does not: " + repr(invented)


def test_the_prose_sweep_finds_the_names_it_is_meant_to_check():
    # Guard the guard: if the pattern or the file list stopped matching, the test
    # above would pass by finding nothing at all. The prose holds 13 tokens today.
    tokens = _prose_tokens()
    assert len(tokens) >= 12, sorted(tokens)
    assert "blue_minus_red_mean" in tokens  # a feature name tutorial 02 uses
    assert "created_by" in tokens  # a key the load/save note names
