"""What an artifact is: the directory that carries an adapter and the run behind it.

Run it from the repository root with ``uv run python examples/lora_artifact.py``.
Nothing here needs MLX, a model or a training run: the directories are written by hand
and read back, so what ``LoRAAdapter`` decides is visible - what makes a directory an
artifact, what "no adapter" means, and what is refused before anything is written.
"""

import tempfile
from pathlib import Path

import orangecontrib.ai_suite.kernel.lora as lora

with tempfile.TemporaryDirectory(prefix="lora-artifact-") as scratch:
    scratch = Path(scratch)

    # 1. An artifact is a directory: the adapter files when there are any, and an
    #    `lora_meta.json` naming the base model, the run's record and the artifact.
    #    Without an adapter it is a way of naming a base model as a model - which is
    #    also what `adapter_enabled = False` means on a `LoRAModel`.
    bare = lora.LoRAAdapter(
        base_model="mlx-community/some-base-model",
        adapter_path=None,
        training={"note": "nothing was trained here"},
        name="the-base-model",
    )
    dest = bare.save(scratch / "unadapted")
    print(f"files       {sorted(path.name for path in dest.iterdir())}")
    back = lora.LoRAAdapter.load(dest)
    print(f"loaded      {back}")
    print(f"            base_model={back.base_model}")
    print(f"            adapter_path={back.adapter_path}")
    print(f"            name={back.name!r} training={back.training}")

    # 2. A directory with no `lora_meta.json` is not an artifact: it is read as a plain
    #    MLX model directory, with the adapter off, because that is what it is.
    plain = scratch / "plain-model"
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    print(f"\nno meta     {lora.LoRAAdapter.load(plain)}")

    # 3. An artifact that names an adapter directory holding no weights is refused
    #    before anything is written, rather than saved as an artifact that cannot load.
    try:
        lora.LoRAAdapter(
            base_model="mlx-community/some-base-model",
            adapter_path=str(scratch),
            training={},
        ).save(scratch / "broken")
    except FileNotFoundError as exc:
        print(f"\nrefused     {exc}")
    print(f"            nothing was written: {not (scratch / 'broken').exists()}")
