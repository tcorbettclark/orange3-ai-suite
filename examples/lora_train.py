"""One LoRA fine-tuning run: the rows, the settings, the events, and the adapter.

Run it from the repository root with ``uv run --extra lora python examples/lora_train.py
<model-dir>``, adding ``--stop`` to end the run early and ``--save DIR`` to keep the
adapter that ``examples/lora_generate.py`` then uses. The kernel's ``lora`` package is
taken as one object and used qualified, so every call below is a name it exports.
"""

import argparse
import tempfile

import orangecontrib.ai_suite.kernel.lora as lora

#: Sentences of one subject. Continued pre-training has no labels: every token in a
#: row is a target, which is what a text corpus means to the trainer.
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
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("base", nargs="?", help="MLX model directory or Hugging Face repo id")
parser.add_argument("--iters", type=int, default=60, help="training iterations")
parser.add_argument("--stop", action="store_true", help="stop at the first training report")
parser.add_argument("--save", metavar="DIR", help="keep the adapter in DIR")
args = parser.parse_args()
if not args.base:
    raise SystemExit(
        "pass a base model: an MLX model directory or a Hugging Face id. "
        "`orange-ai-assets fetch britllm-4bit` prints the cached path."
    )
if not lora.available():
    raise SystemExit(
        "this needs the lora extra: pip install 'orange3-ai-suite[lora]' (macOS on Apple Silicon)"
    )

# 1. The rows. A corpus is texts for continued pre-training or prompt/completion pairs
#    for fine-tuning; the constructor is where a corpus too small to split is refused.
data = lora.LoRAData.from_texts(CORPUS)
print(f"data        {data.format}: {len(data.rows)} rows")

# 2. The settings. `validation_fraction` is the share of rows held out, and
#    `save_every` is how often the trainer writes a checkpoint - the interval a stop
#    has to work with.
options = lora.LoRAOptions(
    iters=args.iters,
    learning_rate=1e-4,
    num_layers=8,
    batch_size=2,
    max_seq_length=256,
    validation_fraction=0.2,
    steps_per_report=10,
    steps_per_eval=30,
    save_every=5,
    seed=0,
)
print(
    f"options     {options.iters} iterations, batch {options.batch_size}, "
    f"{options.validation_fraction:.0%} of the rows held out"
)

with tempfile.TemporaryDirectory(prefix="lora-example-") as scratch:
    # 3. The run. Asking for it validates the settings and splits the rows, so a bad
    #    split fails here, at the call; training starts when the loop asks for the
    #    first event, and `stop()` may be called from any thread at any point.
    run = lora.train_lora(args.base, data, options=options, workdir=scratch, name="example")
    print(f"split       {len(run.train)} train, {len(run.valid)} validation rows")

    print("\nevents")
    try:
        for event in run:
            match event:
                case lora.Params():
                    print(f"  params      {event.fraction:.3f}% trainable ({event.detail})")
                case lora.TrainReport():
                    print(
                        f"  train       iter {event.iteration:4d}  loss {event.loss:.4f}"
                        f"  {event.tokens_per_second} tok/s"
                    )
                    if args.stop:
                        run.stop()  # the loop keeps going until the child is gone
                case lora.ValReport():
                    print(f"  val         iter {event.iteration:4d}  loss {event.loss:.4f}")
    except lora.TrainingStopped as exc:
        raise SystemExit(f"nothing was adopted: {exc}") from exc

    # 4. The artifact. `run.adapter` is set once the iteration ends, however it ended,
    #    and its `training` record says how far the run got and whether a stop cut it
    #    short - a stopped run adopts its newest checkpoint rather than nothing. The
    #    trainer reports an iteration before it saves it, so the checkpoint a stop
    #    adopts is whichever save had already finished, not the iteration the stop
    #    landed on.
    adapter = run.adapter
    record = adapter.training
    print(f"\nadapter     {adapter.name or '(unnamed)'}")
    print(
        f"            {record['iters_completed']} of {record['options']['iters']} iterations,"
        f" stopped={record['stopped']}"
    )
    print(f"            record keys: {sorted(record)}")

    # 5. Where it lives. The run wrote into its own child of `scratch`, which this
    #    `with` deletes; `save` is what turns it into a directory the caller owns.
    if args.save:
        print(f"            saved to {adapter.save(args.save)}")
    else:
        print(f"            in {adapter.adapter_path}, deleted with the scratch:")
        print("            pass --save DIR to keep it")
