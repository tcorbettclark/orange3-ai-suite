# Releasing orange3-ai-suite

The version is written once, in `pyproject.toml`. The tag is made from it, never
typed.

## Rules

1. `[project].version` is the only place a version is written. No `__version__`.
2. The tag is `v` + `uv version --short`. `scripts/release.py` makes both.
3. The `edu-ai-suite` floor moves in the same commit that first uses an API new
   in the lower bound, and the commit names the symbol. It is never raised for
   freshness.
4. Until 1.0 the dependency is bracketed: `>=0.0.1,<0.1.0`. At 1.0 it becomes
   `>=1.0,<2.0`.
5. A published version is never reused. A broken 0.0.2 becomes 0.0.3; the tag
   stays where it is.

## What each number means here

- **patch** — a fix, with no widget, setting or signal change.
- **minor** — a new widget, setting or signal. Before 1.0 this may also rename
  or remove one.
- **major** — 1.0 and after, for a breaking change. A renamed or removed widget
  setting or signal is breaking: `.ows` files persist both.

## The release

```bash
uv run python scripts/release.py check
uv run python scripts/release.py release patch --dry-run
uv run python scripts/release.py release patch --publish
uv run python scripts/release.py verify
```

`check` refuses when the tree is dirty, the branch is not `main`, HEAD is not
`origin/main`, the tag already exists, the lock is stale, a test fails, the
wheel does not install, the dependency is not bracketed, or the lower bound is
not on PyPI.

`release` bumps the version and the lock, re-runs both suites, rebuilds, commits
`release: X.Y.Z`, tags `vX.Y.Z`, and pushes `main` with the tag. `--publish`
uploads both artifacts, waits for the version to appear on PyPI, and installs it
in a fresh environment.

## Raising the floor

`check` proves the bracket is well formed and that the lower bound exists on
PyPI. To prove the floor is real, install exactly it in the Orange interpreter
and run the widget suite:

```bash
uv pip install --python "$ORANGE_PYTHON" "edu-ai-suite==<lower bound>"
QT_QPA_PLATFORM=offscreen "$ORANGE_PYTHON" -m pytest tests/widgets
uv pip install --python "$ORANGE_PYTHON" edu-ai-suite   # back to latest
```

Do this whenever a release raises the lower bound, and before widening the
bracket for a new `edu-ai-suite` minor. `tests/test_release_contract.py` fails
if the bracket is removed or the installed version falls outside it.

## Ordering

When `edu-ai-suite` changes an API used here: it publishes first, this package
raises the floor and adapts in one commit, then this package releases. Never
release against an unpublished dependency.

## If a publish fails

If the tag is pushed and the upload failed, fix the cause and run `uv publish`
again for the same version — it was never published, so it is not being reused.
If the uploaded wheel itself is wrong, do not re-upload it: bump to the next
patch.
