# Ensures the repo root is on sys.path when running bare `pytest`,
# not only `python -m pytest`. Kernel tests must also run under a plain
# Python without Orange installed (REPO-CONTEXT.md §4).