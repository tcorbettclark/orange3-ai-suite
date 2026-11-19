#!/usr/bin/env bash
#
# A LoRA walkthrough: the artifact, a short run, a run worth keeping, the adapter in
# use, and a stop. About twenty seconds on an Apple-silicon laptop once the 1.50 GB
# model is cached; the first run adds the download.
#
#   bash examples/lora_walkthrough.sh
#
# Every Python command below is the one from that example's own docstring, so this can
# be read as a session rather than a script: nothing here reaches past the `lora`
# package's facade.

set -euo pipefail

cd "$(dirname "$0")/.."

#: The interpreter the examples run under: `uv run` when uv is installed, else the
#: repository's own `.venv`. `--extra lora` is not optional - it is what installs
#: mlx-lm, and a plain `uv run` would sync the environment without it.
if command -v uv >/dev/null 2>&1; then
    PY=(uv run --extra lora python)
elif [ -x .venv/bin/python ]; then
    PY=(.venv/bin/python)
else
    echo "needs uv, or a .venv with the lora extra: pip install -e '.[lora]'" >&2
    exit 1
fi

step() {
    printf '\n$ %s\n' "$*"
    "$@"
}

scratch=$(mktemp -d -t lora-walkthrough)
trap 'rm -rf "$scratch"' EXIT

# 1. The artifact on its own. No MLX, no model, no training: this is the directory a
#    run leaves and what `LoRAAdapter` decides about it, so the files that follow make
#    sense when they appear.
step "${PY[@]}" examples/lora_artifact.py

# 2. The model directory. `fetch` prints it on stdout and its progress on stderr, is
#    idempotent, and verifies the SHA-256 of the weights it caches.
model=$("${PY[@]}" -m orangecontrib.ai_suite.kernel.assets fetch britllm-4bit)
if [ ! -d "$model" ]; then
    echo "fetch did not print a model directory: $model" >&2
    exit 1
fi
printf '\nmodel: %s\n' "$model"

# 3. The smallest run that can go wrong: ten iterations, a few seconds, no artifact
#    kept. Enough to see the split, the parameter count and a loss curve.
step "${PY[@]}" examples/lora_train.py "$model" --iters 10

# 4. A run worth keeping. `--save` copies the adapter out of the scratch directory the
#    script owns, which is what makes it an artifact you can point at later.
step "${PY[@]}" examples/lora_train.py "$model" --iters 40 --save "$scratch/adapter"
step ls -1 "$scratch/adapter"

# 5. The adapter in use: one `LoRAModel`, the base model with it off and on, then a
#    save and reload. The artifact names its own base model, so this needs only the
#    directory - not the path from step 2.
step "${PY[@]}" examples/lora_generate.py "$scratch/adapter"

# 6. A stop. The trainer reports an iteration before it saves it, so the checkpoint
#    adopted is whichever save had finished - here iteration 5 of 60, not the 10 the
#    stop landed on. Without `--save` the artifact goes with the scratch.
step "${PY[@]}" examples/lora_train.py "$model" --stop
