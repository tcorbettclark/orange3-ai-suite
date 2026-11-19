#!/usr/bin/env python3
"""Release this package: check the tree, bump the version, tag, push and publish.

check writes no tracked file; it rebuilds dist/. release writes pyproject.toml and
uv.lock, commits release: X.Y.Z, creates the annotated tag vX.Y.Z, pushes main and
the tag, and with --publish uploads the artifacts and verifies them on PyPI.
publish uploads the version already tagged at HEAD, so a tagged release can reach
PyPI later without a second bump. verify compares the project, the tag at HEAD,
and the digests PyPI records for the artifacts in dist/.

The settings are [tool.release] in pyproject.toml, and this file is identical in
both repositories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Filled by main from [tool.release] in pyproject.toml.
CONFIG = {}


def say(message):
    print(f"\n== {message}", flush=True)


def fail(message):
    raise SystemExit(f"release: {message}")


VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(text):
    """Replace ``${NAME}`` and ``${NAME:-default}`` with the environment value."""

    def replace(match):
        value = os.environ.get(match.group(1), match.group(2))
        if value is None:
            fail(f"{match.group(1)} is not set and has no default")
        return value

    return VARIABLE.sub(replace, text)


def load_config():
    """The settings from ``[tool.release]`` in pyproject.toml, plus the project's name."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    settings = project.get("tool", {}).get("release")
    if settings is None:
        fail("add [tool.release] to pyproject.toml")
    unknown = set(settings) - {"smoke", "tests", "test_env", "dependency"}
    if unknown:
        fail(f"unknown [tool.release] keys: {', '.join(sorted(unknown))}")
    for key in ("smoke", "tests"):
        if key not in settings:
            fail(f"[tool.release] needs {key}")
    if not isinstance(settings["smoke"], str):
        fail("[tool.release] smoke must be a Python one-liner")
    tests = settings["tests"]
    if not isinstance(tests, list) or not tests:
        fail("[tool.release] tests must be a non-empty list of command arrays")
    for command in tests:
        if not isinstance(command, list) or not command:
            fail("[tool.release] tests must be a non-empty list of command arrays")
        if not all(isinstance(part, str) for part in command):
            fail("[tool.release] tests: every part of a command must be a string")
    test_env = settings.get("test_env", {})
    if not isinstance(test_env, dict) or not all(
        isinstance(item, str) for item in test_env.values()
    ):
        fail("[tool.release] test_env must be a table of strings")
    return {
        "package": project["project"]["name"],
        "smoke": settings["smoke"],
        "tests": [[expand(part) for part in command] for command in tests],
        "test_env": {name: expand(value) for name, value in test_env.items()},
        "dependency": settings.get("dependency"),
    }


def run(command, env=None, show=True, capture=False, cwd=None):
    if show:
        print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd or ROOT, env=env, text=True, capture_output=capture)
    if result.returncode != 0:
        if capture:
            print(result.stdout, result.stderr)
        fail(f"command failed: {' '.join(command)}")
    return result.stdout.strip() if capture else ""


def git(*arguments):
    run(["git", *arguments])


def git_read(*arguments):
    return run(["git", *arguments], show=False, capture=True)


def current_version():
    return run(["uv", "version", "--short"], show=False, capture=True)


def pypi_releases():
    url = f"https://pypi.org/pypi/{CONFIG['package']}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return set(json.load(response)["releases"])
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return set()
        raise


def pypi_digests(version):
    url = f"https://pypi.org/pypi/{CONFIG['package']}/{version}/json"
    with urllib.request.urlopen(url, timeout=15) as response:
        return {item["filename"]: item["digests"]["sha256"] for item in json.load(response)["urls"]}


def check_published(version):
    """PyPI serves the artifacts in dist/ for this version."""
    say(f"the published artifacts {version}")
    deadline = time.time() + 120
    while time.time() < deadline:
        if version in pypi_releases():
            break
        time.sleep(3)
    else:
        fail(f"{version} did not appear on PyPI within 120 seconds")
    normalized = CONFIG["package"].replace("-", "_")
    artifacts = sorted((ROOT / "dist").glob(f"{normalized}-{version}*"))
    if not artifacts:
        fail(f"dist/ holds no {version} artifacts; run the build")
    served = pypi_digests(version)
    for artifact in artifacts:
        if artifact.name not in served:
            fail(f"PyPI does not serve {artifact.name}")
        if served[artifact.name] != hashlib.sha256(artifact.read_bytes()).hexdigest():
            fail(f"PyPI serves a different {artifact.name}")


def check_tree():
    say("the tree")
    git("fetch", "origin")
    branch = git_read("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        fail(f"release from main, not {branch}")
    if git_read("status", "--porcelain"):
        fail("the working tree is not clean")
    if git_read("rev-parse", "HEAD") != git_read("rev-parse", "origin/main"):
        fail("HEAD is not origin/main; push or pull first")


def check_tag(version):
    if git_read("tag", "--list", f"v{version}"):
        fail(f"tag v{version} already exists; bump instead of reusing it")


def run_tests():
    say("the tests")
    env = {**os.environ, **CONFIG["test_env"]}
    for command in CONFIG["tests"]:
        if command[0].startswith("/") and not pathlib.Path(command[0]).exists():
            fail(f"{command[0]} does not exist")
        run(command, env=env)


def build():
    say("the artifact")
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    run(["uv", "build"])
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    if len(wheels) != 1:
        fail(f"expected one wheel in dist/, found {len(wheels)}")
    return wheels[0]


def smoke(install):
    say("a fresh environment")
    with tempfile.TemporaryDirectory() as directory:
        venv = pathlib.Path(directory) / "venv"
        run(["uv", "venv", "--quiet", str(venv)])
        python = venv / "bin" / "python"
        run(["uv", "pip", "install", "--quiet", "--python", str(python), str(install)])
        run([str(python), "-c", CONFIG["smoke"]], cwd=directory)


def dependency_specs():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    dependency = CONFIG["dependency"]
    specs = [item for item in project["dependencies"] if item.startswith(dependency)]
    for extra in project.get("optional-dependencies", {}).values():
        specs += [item for item in extra if item.startswith(dependency)]
    return specs


def check_dependency():
    dependency = CONFIG["dependency"]
    if not dependency:
        return
    say(f"the floor on {dependency}")
    pattern = re.compile(
        rf"{re.escape(dependency)}(?:\[[a-z]+\])?>=(\d+\.\d+\.\d+),<(\d+\.\d+\.\d+)"
    )
    bounds = set()
    for spec in dependency_specs():
        match = pattern.fullmatch(spec)
        if not match:
            fail(f"{spec} is not a >=X.Y.Z,<X.Y.Z bracket")
        bounds.add(
            tuple(tuple(int(number) for number in group.split(".")) for group in match.groups())
        )
    if not bounds:
        fail(f"no {dependency} requirement found")
    if len(bounds) != 1:
        fail("the base and extra requirements use different brackets")
    lower, upper = bounds.pop()
    if not lower < upper:
        fail("the lower bound is not below the upper bound")
    lower_text = ".".join(str(part) for part in lower)
    if lower_text not in pypi_releases():
        fail(f"{dependency} {lower_text} is not on PyPI")


def cmd_check():
    check_tree()
    version = current_version()
    tagged = f"v{version}" in git_read("tag", "--list").split()
    published = version in pypi_releases()
    if tagged and not published:
        fail(f"{version} is tagged but not on PyPI; run publish, do not bump")
    say(f"version: {version} (tagged: {tagged}, on PyPI: {published})")
    say("the lock")
    run(["uv", "lock", "--check"])
    run_tests()
    wheel = build()
    smoke(wheel)
    check_dependency()
    say(f"ready: {CONFIG['package']} {version}")


def publish_version(version):
    say(f"publish {version}")
    run(["uv", "publish", "--check-url", "https://pypi.org/simple/"])
    check_published(version)
    say(f"published {CONFIG['package']} {version}")


def cmd_release(kind, publish, dry_run):
    cmd_check()
    version = current_version()
    following = run(
        ["uv", "version", "--bump", kind, "--dry-run", "--short"], show=False, capture=True
    )
    check_tag(following)
    say(f"bump: {version} -> {following}")
    if dry_run:
        print("dry run: nothing was committed, tagged or published", flush=True)
        return
    run(["uv", "version", "--bump", kind, "--no-sync"])
    written = current_version()
    if written != following:
        fail(f"uv version wrote {written}, expected {following}")
    run_tests()
    wheel = build()
    smoke(wheel)
    say("the commit and the tag")
    run(["git", "add", "pyproject.toml", "uv.lock"])
    run(["git", "commit", "-m", f"release: {written}"])
    run(["git", "tag", "-a", f"v{written}", "-m", f"{CONFIG['package']} {written}"])
    run(["git", "push", "origin", "main", f"v{written}"])
    if publish:
        publish_version(written)
    else:
        say(f"tagged {written}; run publish when you want it on PyPI")


def cmd_publish():
    version = current_version()
    say(f"publish {CONFIG['package']} {version}")
    check_tree()
    if f"v{version}" not in git_read("tag", "--points-at", "HEAD").split():
        fail(f"HEAD is not tagged v{version}; publish from the release commit")
    if not git_read("ls-remote", "--tags", "origin", f"v{version}"):
        fail(f"tag v{version} is not on origin; push it before publishing")
    if version in pypi_releases():
        fail(f"{version} is already on PyPI")
    wheel = build()
    smoke(wheel)
    publish_version(version)


def cmd_verify():
    version = current_version()
    say(f"verify {CONFIG['package']} {version}")
    if f"v{version}" not in git_read("tag", "--points-at", "HEAD").split():
        fail(f"HEAD is not tagged v{version}")
    normalized = CONFIG["package"].replace("-", "_")
    if not list((ROOT / "dist").glob(f"{normalized}-{version}-*.whl")):
        fail(f"dist/ holds no {version} wheel; run the build")
    check_published(version)
    say(f"verified: {CONFIG['package']} {version}")


def main():
    global CONFIG
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="run every pre-release check; write no tracked file")
    release = commands.add_parser("release", help="bump, test, commit, tag and push")
    release.add_argument("kind", choices=["patch", "minor", "major"])
    release.add_argument("--publish", action="store_true", help="upload and verify on PyPI")
    release.add_argument(
        "--dry-run", action="store_true", help="print the next version; write nothing"
    )
    commands.add_parser("publish", help="upload the version already tagged at HEAD")
    commands.add_parser("verify", help="compare the project, the tag, PyPI and dist/")
    arguments = parser.parse_args()
    CONFIG = load_config()
    if arguments.command == "check":
        cmd_check()
    elif arguments.command == "release":
        cmd_release(arguments.kind, arguments.publish, arguments.dry_run)
    elif arguments.command == "publish":
        cmd_publish()
    else:
        cmd_verify()


if __name__ == "__main__":
    main()
