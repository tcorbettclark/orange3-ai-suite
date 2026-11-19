#!/usr/bin/env python3
"""Release this package: check the tree, bump the version, tag, push and publish.

``check`` writes no tracked file; it rebuilds ``dist/``. ``release`` writes
``pyproject.toml`` and ``uv.lock``, commits ``release: X.Y.Z``, creates the
annotated tag ``vX.Y.Z``, pushes ``main`` and the tag, and with ``--publish``
uploads the artifacts and verifies them on PyPI. ``verify`` compares the project,
the tag at HEAD, PyPI and ``dist/``.
"""

from __future__ import annotations

import argparse
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

ORANGE_PYTHON = os.environ.get("ORANGE_PYTHON", "/Applications/Orange.app/Contents/MacOS/python")

CONFIG = {
    "package": "orange3-ai-suite",
    "tests": [
        ["uv", "run", "pytest"],
        [ORANGE_PYTHON, "-m", "pytest", "tests/widgets"],
    ],
    "test_env": {"QT_QPA_PLATFORM": "offscreen"},
    "smoke": "import orangecontrib.ai_suite as add_on; print(add_on.__file__)",
    "dependency": "edu-ai-suite",
    "publish_gate": False,
}


def say(message):
    print(f"\n== {message}", flush=True)


def fail(message):
    raise SystemExit(f"release: {message}")


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
            fail(f"{command[0]} does not exist; set ORANGE_PYTHON")
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


def print_gate(version, wheel):
    if not CONFIG["publish_gate"]:
        return
    say("the downstream gate")
    print(
        "Before publishing a minor, run the add-on's widget suite against this\n"
        "candidate wheel in the Orange interpreter, then restore the published one:\n\n"
        f"  uv pip install --python {ORANGE_PYTHON} --force-reinstall --no-deps {wheel}\n"
        f"  (cd ../orange3-ai-suite && QT_QPA_PLATFORM=offscreen {ORANGE_PYTHON}"
        " -m pytest tests/widgets)\n"
        f"  uv pip install --python {ORANGE_PYTHON} {CONFIG['package']}=={version}\n\n"
        "Pass --gate-confirmed to --publish when it is green."
    )


def cmd_check():
    check_tree()
    version = current_version()
    tagged = f"v{version}" in git_read("tag", "--list").split()
    published = version in pypi_releases()
    if tagged and not published:
        fail(f"{version} is tagged but not on PyPI; publish it rather than bumping")
    say(f"version: {version} (tagged: {tagged}, on PyPI: {published})")
    say("the lock")
    run(["uv", "lock", "--check"])
    run_tests()
    wheel = build()
    smoke(wheel)
    check_dependency()
    print_gate(version, wheel)
    say(f"ready: {CONFIG['package']} {version}")


def publish_version(version):
    say(f"publish {version}")
    run(["uv", "publish"])
    deadline = time.time() + 120
    while time.time() < deadline:
        if version in pypi_releases():
            break
        time.sleep(3)
    else:
        fail("the version did not appear on PyPI within 120 seconds")
    smoke(f"{CONFIG['package']}=={version}")
    say(f"published {CONFIG['package']} {version}")


def cmd_release(kind, publish, gate_confirmed, dry_run):
    cmd_check()
    version = current_version()
    following = run(
        ["uv", "version", "--bump", kind, "--dry-run", "--short"], show=False, capture=True
    )
    check_tag(following)
    say(f"bump: {version} -> {following}")
    if publish and CONFIG["publish_gate"] and not gate_confirmed:
        fail("--publish needs --gate-confirmed for this package")
    if dry_run:
        print("dry run: nothing was committed, tagged or published", flush=True)
        return
    run(["uv", "version", "--bump", kind, "--no-sync"])
    run_tests()
    wheel = build()
    smoke(wheel)
    say("the commit and the tag")
    run(["git", "add", "pyproject.toml", "uv.lock"])
    run(["git", "commit", "-m", f"release: {following}"])
    run(["git", "tag", "-a", f"v{following}", "-m", f"{CONFIG['package']} {following}"])
    run(["git", "push", "origin", "main", f"v{following}"])
    if publish:
        publish_version(following)


def cmd_verify():
    version = current_version()
    say(f"verify {CONFIG['package']} {version}")
    if f"v{version}" not in git_read("tag", "--points-at", "HEAD").split():
        fail(f"HEAD is not tagged v{version}")
    if version not in pypi_releases():
        fail(f"{version} is not on PyPI")
    normalized = CONFIG["package"].replace("-", "_")
    if not list((ROOT / "dist").glob(f"{normalized}-{version}-*.whl")):
        fail(f"dist/ holds no {version} wheel; run the build")
    say(f"verified: {CONFIG['package']} {version}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="run every pre-release check; write no tracked file")
    release = commands.add_parser("release", help="bump, test, commit, tag and push")
    release.add_argument("kind", choices=["patch", "minor", "major"])
    release.add_argument("--publish", action="store_true", help="upload and verify on PyPI")
    release.add_argument(
        "--gate-confirmed", action="store_true", help="the downstream suite passed"
    )
    release.add_argument(
        "--dry-run", action="store_true", help="print the next version; write nothing"
    )
    commands.add_parser("verify", help="compare the project, the tag, PyPI and dist/")
    arguments = parser.parse_args()
    if arguments.command == "check":
        cmd_check()
    elif arguments.command == "release":
        cmd_release(arguments.kind, arguments.publish, arguments.gate_confirmed, arguments.dry_run)
    else:
        cmd_verify()


if __name__ == "__main__":
    main()
