"""The training child: the process mlx-lm trains in, reporting through :mod:`mlx_protocol`.

It runs as ``python -m ...lora.mlx_child <json spec>``: the parent passes a JSON
:class:`~mlx_spec.TrainingSpec` and the child rebuilds it by name, so a field one side
does not know fails here. Every line on stdout that is not a marked event is mlx-lm's
human log, which the parent keeps for a failure message.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from typing import Any

from .mlx_protocol import Params, TrainReport, ValReport, encode
from .mlx_spec import TrainingSpec


def emit(event: Params | TrainReport | ValReport) -> None:
    """Write one structured event, flushed: the parent reads lines as they arrive."""
    sys.stdout.write(encode(event) + "\n")
    sys.stdout.flush()


def apply_spec(args: Any, spec: TrainingSpec) -> None:
    """Overwrite the fields of an mlx-lm argument namespace with the run's specification.

    The namespace comes from mlx-lm's own parser, so every field the spec does not name
    keeps mlx-lm's default.
    """
    for field in fields(TrainingSpec):
        setattr(args, field.name, getattr(spec, field.name))
    args.train = True
    args.test = False
    args.fine_tune_type = "lora"
    # mlx-lm names the checkpoint interval differently from our options.
    args.save_every = spec.save_every


def emit_params(model: Any) -> None:
    """The trainable-parameter counts, as data: one metric before training starts."""
    from mlx.utils import tree_flatten
    from mlx_lm.utils import get_total_parameters

    total = get_total_parameters(model)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    emit(
        Params(
            fraction=100.0 * trainable / total,
            detail=f"{trainable / 1e6:.3f}M/{total / 1e6:.3f}M",
        )
    )


def install_params_hook(lora_module: Any) -> None:
    """Replace mlx-lm's trainable-parameter printer with one that also emits the counts.

    The printer is reached through a module global in ``mlx_lm.lora``, so a missing one
    raises here rather than dropping the metric silently.
    """
    original = getattr(lora_module, "print_trainable_parameters", None)
    if original is None:
        raise RuntimeError(
            "mlx-lm no longer reports trainable parameters through "
            "mlx_lm.lora.print_trainable_parameters; emit_params needs a new hook"
        )

    def hook(model: Any) -> None:
        original(model)  # keep mlx-lm's own line in the human log
        emit_params(model)

    lora_module.print_trainable_parameters = hook


def emit_metrics(training_callback: type) -> Any:
    """A callback on the given base class that emits one event per loss report."""

    class Emitter(training_callback):  # type: ignore[misc, valid-type]
        def on_train_loss_report(self, info: dict) -> None:
            emit(
                TrainReport(
                    iteration=int(info["iteration"]),
                    loss=float(info["train_loss"]),
                    tokens_per_second=int(info["tokens_per_second"]),
                )
            )

        def on_val_loss_report(self, info: dict) -> None:
            # mlx-lm reports the iteration *before* the evaluation it just ran
            # ("iteration": it - 1 in tuner/trainer.py), so +1 keeps both curves
            # on the same axis as the training reports.
            emit(
                ValReport(
                    iteration=int(info["iteration"]) + 1,
                    loss=float(info["val_loss"]),
                )
            )

    return Emitter()


def main(argv: list[str] | None = None) -> int:
    """Train one run from the JSON specification in ``argv[1]``."""
    spec = TrainingSpec(**json.loads((sys.argv if argv is None else argv)[1]))

    import mlx_lm.lora as lora
    from mlx_lm import load
    from mlx_lm.tuner.callbacks import TrainingCallback

    # mlx-lm's own defaults: a field it adds tomorrow arrives with its default
    # rather than as an AttributeError halfway through a run.
    args = lora.build_parser().parse_args([])
    apply_spec(args, spec)
    install_params_hook(lora)

    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )
    train_set, valid_set, _test_set = lora.load_dataset(args, tokenizer)
    lora.train_model(args, model, train_set, valid_set, emit_metrics(TrainingCallback))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
