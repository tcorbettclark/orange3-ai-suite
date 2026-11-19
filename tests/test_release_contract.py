"""The dependency this add-on publishes is bracketed, and the installed version satisfies it."""

import importlib.metadata
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPENDENCY = "edu-ai-suite"
BRACKET = re.compile(
    rf"{re.escape(DEPENDENCY)}(?:\[[a-z]+\])?>=(\d+\.\d+\.\d+),<(\d+\.\d+\.\d+)"
)


def _requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [item for item in project["dependencies"] if item.startswith(DEPENDENCY)]
    for extra in project.get("optional-dependencies", {}).values():
        requirements += [item for item in extra if item.startswith(DEPENDENCY)]
    return requirements


def _version(text):
    return tuple(int(part) for part in text.split("."))


def test_every_requirement_is_a_bracket():
    requirements = _requirements()
    assert requirements
    for requirement in requirements:
        assert BRACKET.fullmatch(requirement), requirement


def test_the_bounds_are_ordered_and_shared():
    matches = [BRACKET.fullmatch(item) for item in _requirements()]
    assert all(matches), _requirements()
    bounds = {match.groups() for match in matches}
    assert len(bounds) == 1, bounds
    lower, upper = bounds.pop()
    assert _version(lower) < _version(upper)


def test_the_installed_dependency_satisfies_the_bracket():
    lower, upper = BRACKET.fullmatch(_requirements()[0]).groups()
    installed = _version(importlib.metadata.version(DEPENDENCY))
    assert _version(lower) <= installed < _version(upper), installed
