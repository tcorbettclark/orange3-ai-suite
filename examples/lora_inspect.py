"""What a directory is, as far as LoRA is concerned: an artifact, and what it says.

Run it from the repository root with ``uv run python examples/lora_inspect.py <dir>``,
pointing at what ``examples/lora_train.py --save`` wrote - or at a plain MLX model
directory, which it will tell you is not an artifact. No MLX and no base model are
needed: everything below is read from the artifact's own files.

The three are ``lora_meta.json`` (the base model, the run's settings, how far it got),
``adapter_config.json`` (mlx-lm's description of the adapter) and
``adapters.safetensors``, whose layout is eight bytes of header length, then that many
bytes of JSON naming each tensor, its dtype and its shape, then the weights.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import orangecontrib.ai_suite.kernel.lora as lora


def tensor_header(weights: Path) -> dict:
    """The JSON header of a safetensors file: tensor name -> {dtype, shape, offsets}."""
    with weights.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def human(count: float) -> str:
    """A count as a short string: 3104768 -> '3.105M'."""
    for unit, size in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if count >= size:
            return f"{count / size:.3f}{unit}"
    return str(count)


def directory_bytes(directory: Path) -> int:
    """How many bytes of files a directory holds, following the links a Hub cache uses."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def sizes(directory: Path) -> list[str]:
    """The files in a directory, smallest first, as 'name  size'."""
    found = [(path.stat().st_size, path.name) for path in directory.iterdir() if path.is_file()]
    return [f"{name}  {human(size)}B" for size, name in sorted(found)]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", help="an artifact directory, or a plain MLX model directory")
args = parser.parse_args()

directory = Path(args.path)
adapter = lora.LoRAAdapter.load(directory)

if not (directory / "lora_meta.json").is_file():
    print(f"{directory}  {human(directory_bytes(directory))}B")
    print("  no lora_meta.json, so this is not an artifact: it is read as the base model")
    print("  itself, with the adapter off.")
    if (directory / "adapter_config.json").is_file():
        print("  adapter_config.json is here all the same: a run's adapter directory looks")
        print("  like this before `save`, and only `save` adds the record.")
    raise SystemExit(0)

record = adapter.training
print(f"artifact    {directory}")
print(f"  name        {adapter.name or '(unnamed)'}")

# The adapter is a delta, so none of the model it adjusts is in this directory: the
# recorded name is the whole of the dependency, and it is resolved from wherever this
# is run rather than from where the artifact was made.
base = Path(adapter.base_model)
print("\ndependency  the model this adapter adjusts; none of it is in this directory")
print(f"  recorded    {adapter.base_model}")
if base.is_dir():
    print(f"  found       a directory, {human(directory_bytes(base))}B on disk")
elif base.exists():
    print(f"  found       not a directory: {base.resolve()}")
else:
    print(f"  missing     no such directory from here ({base.resolve()})")
    print("              mlx-lm looks the name up as a Hugging Face repo id instead")

print(
    f"\nrun         {record.get('iters_completed')} of {record.get('options', {}).get('iters')}"
    f" iterations, stopped {record.get('stopped')}"
)
options = record.get("options", {})
print(
    f"  options     lr {options.get('learning_rate')}, batch {options.get('batch_size')},"
    f" max_seq {options.get('max_seq_length')}, val {options.get('validation_fraction')},"
    f" seed {options.get('seed')}"
)
print(
    f"  cadence     report every {options.get('steps_per_report')}, evaluate every"
    f" {options.get('steps_per_eval')}, checkpoint every {options.get('save_every')}"
)

# The paths the run recorded are in the scratch directory the caller owned, so they
# outlive it or not: this is what says whether the artifact is self-contained.
for key in ("run_dir",):
    where = record.get(key)
    if where:
        print(f"  {key:11s} {where}  ({'present' if Path(where).exists() else 'gone'})")
data = record.get("adapter_config", {}).get("data")
if data:
    print(f"  data        {data}  ({'present' if Path(data).exists() else 'gone'})")

if adapter.adapter_path is None:
    print("\nadapter     none: this artifact names a base model and no adapter")
    raise SystemExit(0)

adapter_dir = Path(adapter.adapter_path)
weights = adapter_dir / "adapters.safetensors"
config = record.get("adapter_config", {})
lora_parameters = config.get("lora_parameters") or {}
print(f"\nadapter     {', '.join(sizes(adapter_dir))}")
print(
    f"  config      {config.get('fine_tune_type')}, rank {lora_parameters.get('rank')},"
    f" dropout {lora_parameters.get('dropout')}, scale {lora_parameters.get('scale')}"
)
print(f"  masking     prompt {config.get('mask_prompt')}")

if not weights.is_file():
    print(f"  refused     no {weights.name} here, so nothing can be loaded from it")
    raise SystemExit(1)

# What the tensors say, rather than what the config says they should: the narrow
# dimension of a factor is the rank it was trained at, and the digits in the names are
# the blocks it reached.
header = tensor_header(weights)
values = sum(math.prod(tensor["shape"]) for tensor in header.values())
ranks = sorted({min(tensor["shape"]) for tensor in header.values()})
blocks = sorted({int(part) for name in header for part in name.split(".") if part.isdigit()})
modules = sorted({name.split(".")[-2] for name in header})
dtypes = sorted({tensor["dtype"] for tensor in header.values()})

print(f"  tensors     {len(header)} ({', '.join(dtypes)}), {human(values)} values")
print(f"  rank        {', '.join(map(str, ranks))}")
span = f"{blocks[0]}-{blocks[-1]}" if blocks else "none"
print(f"  blocks      {len(blocks)} ({span})")
print(f"  modules     {', '.join(modules)}")
print(
    f"  check       config says {config.get('num_layers')} layers at rank"
    f" {lora_parameters.get('rank')}; the file holds {len(blocks)} blocks at rank"
    f" {', '.join(map(str, ranks))}"
)
