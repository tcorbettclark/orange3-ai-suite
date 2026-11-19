#!/usr/bin/env bash
#
# A LoRA walkthrough: the model, a short run, a run worth keeping, the artifact that
# records it, that adapter in use, and a stop. About twenty seconds on an
# Apple-silicon laptop once the 1.50 GB model is cached; the first run adds the
# download.
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

steps=7
step=0
print_step() {
    step=$((step + 1))
    local title=$1
    shift
    printf '\n'
    printf -- '--------------------------------------------------------------------------\n'
    printf '%d/%d  %s\n' "$step" "$steps" "$title"
    printf -- '--------------------------------------------------------------------------\n'
    printf '$ %s\n' "$*"
}
run_step() {
    print_step "$@"   # takes the title off its own copy of the arguments
    shift
    "$@"
}

scratch=$(mktemp -d -t lora-walkthrough)
trap 'rm -rf "$scratch"' EXIT

# `fetch` prints the directory on stdout and its progress on stderr, so its stdout is
# captured rather than run as a step; it is idempotent and verifies the SHA-256 of
# what it caches.
fetch=("${PY[@]}" -m orangecontrib.ai_suite.kernel.assets fetch britllm-4bit)
print_step "the model directory" "${fetch[@]}"
model=$("${fetch[@]}")
if [ ! -d "$model" ]; then
    echo "fetch did not print a model directory: $model" >&2
    exit 1
fi
printf 'model: %s\n' "$model"

# Point the viewer at the model directory and it reports that: no meta.json, so
# nothing that claims to be a trained artifact.
run_step "the base model directory is not an artifact" \
    "${PY[@]}" examples/lora_inspect.py "$model"

# The smallest run that can go wrong: ten iterations, a few seconds, no artifact kept.
run_step "ten iterations, nothing kept" \
    "${PY[@]}" examples/lora_train.py "$model" --iters 10

# A run worth keeping. `--save` copies the adapter out of the scratch directory the
# script owns, which is what makes it an artifact you can point at later.
run_step "forty iterations, saved" \
    "${PY[@]}" examples/lora_train.py "$model" --iters 40 --save "$scratch/adapter"

# And now it is an artifact: base model, rank, which blocks it reached, how many
# values it holds, and how far the run got - all read from the files, no MLX.
run_step "the artifact records the run" \
    "${PY[@]}" examples/lora_inspect.py "$scratch/adapter"

# The adapter in use: one `LoRAModel`, the base model with it off and on, then a save
# and reload. The artifact names its own base model, so this needs only the directory.
run_step "the adapter in use" \
    "${PY[@]}" examples/lora_generate.py "$scratch/adapter"

# A stop. The trainer reports an iteration before it saves it, so the checkpoint
# adopted is whichever save had finished - here iteration 5 of 60, not the 10 the stop
# landed on. Without `--save` the artifact goes with the scratch.
run_step "a stop, and the files it leaves" \
    "${PY[@]}" examples/lora_train.py "$model" --stop
