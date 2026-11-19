"""One LoRA training run at the MLX boundary: rows in, events out, a directory back.

Run it from the repository root with
``uv run --extra lora python examples/lora_mlx_run.py <model-dir>``. This is the layer
below the ``kernel.lora`` façade, so each piece stands alone: the jsonl mlx-lm reads,
the specification a run is, the events the training child reports, and the adapter
directory the run leaves behind. ``examples/lora_mlx_generate.py`` picks that directory
up and uses it.
"""

import argparse
import json
import tempfile
from dataclasses import fields
from pathlib import Path

from orangecontrib.ai_suite.kernel.lora import (
    mlx_layout,
    mlx_process,
    mlx_protocol,
    mlx_runtime,
    mlx_spec,
)

# A corpus of one-sentence units. Continued pre-training has no labels: every token
# in a row is a target, which is what `mask_prompt = False` below means.
CORPUS = [
    "Cirrus clouds are thin and wispy, and they sit high in the troposphere.",
    "Cirrus are made of ice crystals rather than water droplets.",
    "Contrails often spread into cirrus-like sheets.",
    "Cirrostratus form a veil that dims the sun without hiding it.",
    "A halo around the sun is a sign of cirrostratus.",
    "Altocumulus are mid-level patches, often in rows.",
    "Altostratus are a grey sheet that can thicken into rain.",
    "Stratocumulus are the commonest cloud over the ocean.",
    "Stratocumulus arrive as a layer of rounded rolls with gaps.",
    "Stratus is a featureless grey layer close to the ground.",
    "Fog is stratus that has reached the surface.",
    "Nimbostratus bring steady rain and hide the sun entirely.",
    "Cumulus have flat bases and cauliflower tops.",
    "Fair-weather cumulus are wider than they are tall.",
    "Cumulus congestus grow tall enough to shower.",
    "Cumulonimbus are the storm clouds, with an anvil at the top.",
    "The anvil of a cumulonimbus is made of ice.",
    "Clouds form where rising air cools to its dew point.",
    "Condensation needs a nucleus, such as a dust or salt particle.",
    "A cloud's base marks the height of the lifting condensation level.",
    "Warm air holds more water vapour than cold air.",
    "Adiabatic cooling explains why rising air reaches saturation.",
    "Clouds are named by genus, species and variety.",
    "The height of a cloud is called its level, not its altitude band.",
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("base", nargs="?", help="MLX model directory or Hugging Face repo id")
parser.add_argument("--iters", type=int, default=60)
args = parser.parse_args()
if not args.base:
    raise SystemExit(
        "pass a base model: an MLX model directory or a Hugging Face id. "
        "`orange-ai-assets fetch britllm-4bit` prints the cached path."
    )
if not mlx_runtime.available():
    raise SystemExit(
        "this needs the lora extra: pip install 'orange3-ai-suite[lora]' (macOS on Apple Silicon)"
    )

with tempfile.TemporaryDirectory(prefix="lora-example-") as scratch:
    scratch = Path(scratch)

    # 1. Staging. The boundary owns the shape of the files mlx-lm reads; which rows go
    #    to which side is the caller's decision, so it is handed two lists. mlx-lm
    #    fails on a valid split smaller than one batch, so batch_size is a floor here.
    train, valid = CORPUS[:-4], CORPUS[-4:]
    data_dir = mlx_layout.write_dataset(scratch / "data", "text", train, valid)
    print("dataset   ", data_dir)
    for line in (data_dir / "train.jsonl").read_text().splitlines()[:2]:
        print("           ", line)
    print(f"            {len(train)} train rows, {len(valid)} valid rows")

    # 2. A specification. Every setting a training call takes, as one frozen object, in
    #    mlx-lm's own names - it is what the child rebuilds from the parent's JSON.
    spec = mlx_spec.TrainingSpec(
        model=args.base,
        data=str(data_dir),
        adapter_path=str(scratch / "adapter"),
        iters=args.iters,
        learning_rate=1e-4,
        num_layers=8,
        batch_size=2,
        max_seq_length=256,
        steps_per_report=10,
        steps_per_eval=30,
        save_every=30,
        seed=0,
        mask_prompt=False,  # raw text: nothing is a prompt, every token is a target
    )
    print(f"\nspec       {len(fields(spec))} fields; mask_prompt={spec.mask_prompt}")

    # 3. The run. Constructing it starts nothing: the child process exists only while
    #    the loop below is running, and the first event is the parameter count, before
    #    any loss. A stop is `run.stop()` from any thread, including during this loop.
    run = mlx_process.Run(spec)
    print("\nevents")
    for event in run:
        match event:
            case mlx_protocol.Params():
                print(f"  params     {event.fraction:.3f}% of the model trainable ({event.detail})")
            case mlx_protocol.TrainReport():
                print(
                    f"  train      iter {event.iteration:4d}  loss {event.loss:.4f}"
                    f"  {event.tokens_per_second} tok/s"
                )
            case mlx_protocol.ValReport():
                print(f"  val        iter {event.iteration:4d}  loss {event.loss:.4f}")

    # 4. The directory. `outcome` says whether a stop ended the run early. mlx-lm writes
    #    a numbered checkpoint every `save_every` iterations and the final adapter at the
    #    end, so `adapter_weights` answers "which file is the adapter" either way.
    print(f"\noutcome    stopped={run.outcome.stopped}")
    weights = mlx_layout.adapter_weights(spec.adapter_path)
    if weights is None:
        print("           no checkpoint was written: a stop needs `save_every` iterations first")
    else:
        print(f"adapter    {weights.name}  (iteration {mlx_layout.checkpoint_iteration(weights)})")
    print(f"files      {sorted(p.name for p in Path(spec.adapter_path).iterdir())}")

    # 5. The config is mlx-lm's record of how the adapter was built, written once at the
    #    start of the run. It, and the adapter, are all a caller has to keep: the rest of
    #    `scratch` is one run's working files.
    config = Path(spec.adapter_path) / mlx_layout.ADAPTER_CONFIG
    if config.exists():
        written = json.loads(config.read_text())
        print(f"config     {len(written)} keys, e.g. {sorted(written)[:4]}")
