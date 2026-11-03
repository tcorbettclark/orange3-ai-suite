# orange3-nn-suite

Teaching neural networks and LLMs inside [Orange](https://orangedatamining.com/):
interpretable image features, tiny Keras language models, and MLX LoRA
fine-tuning — one add-on, one "Neural Networks" toolbox category, built on a
pure-Python kernel (`nn_kernel/`) with no Orange or Qt imports.

**Status:** design-stage + kernel phase 1. The image feature extractor kernel
(groups A–F, 36 features) is implemented and tested; the widget layer and the
language-model kernel are not yet started. Design lives in `spec/` — read
`spec/REPO-CONTEXT.md` first.

## Development

```bash
uv sync --extra lm --no-install-package Orange3
# ^ Orange3 is a runtime dependency (the add-on installer needs it) but must
# NOT be installed into the dev venv: kernel tests must run without Orange
# (REPO-CONTEXT §4), and widget tests run under the Orange interpreter:
uv pip install --python /Applications/Orange.app/Contents/MacOS/python \
    --no-deps -e . --no-build-isolation   # one-time: hatchling/editables first
uv run pytest                          # kernel tests, plain Python
/Applications/Orange.app/Contents/MacOS/python -m pytest -q  # widget tests, offscreen
```

Install for Orange (macOS Orange.app bundle), per `spec/REPO-CONTEXT.md` §4:

```bash
uv pip install --python /Applications/Orange.app/Contents/MacOS/python \
    --no-deps dist/orange3_nn_suite-*.whl
```

## License

MIT. Caution: the community add-on that inspired the widget pattern
(`orange3-imagenets`) is CC-BY-NC-SA — **pattern inspiration only, never code
copying** (`spec/REPO-CONTEXT.md` §3).