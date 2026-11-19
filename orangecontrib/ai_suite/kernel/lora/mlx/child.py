"""The training child: our code, so the parent reads a format we own.

mlx-lm's progress output is an ANSI-coloured table meant for a person at a
terminal. Rather than parse it, this module calls mlx-lm's own Python API -
``build_parser`` for the current defaults, ``load`` and ``load_dataset`` to build
what training needs, ``train_model`` with a ``TrainingCallback`` - and writes one
JSON object per line on stdout using the format owned by :mod:`protocol`.
Everything else on stdout is mlx-lm's human log, which the parent forwards to its
tail untouched.

It runs as ``python -m ...mlx.child <json spec>``: the parent passes
:class:`~train.TrainingSpec` as JSON and this module rebuilds it by name, so a
field added on one side and not the other fails here rather than being ignored.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from typing import Any

from .protocol import encode
from .train import TrainingSpec


def emit(metric: dict[str, Any]) -> None:
    """Write one structured metric, flushed: the parent reads lines as they arrive."""
    sys.stdout.write(encode(metric) + "\n")
    sys.stdout.flush()


def apply_spec(args: Any, spec: TrainingSpec) -> None:
    """Fill an mlx-lm argument namespace from the run specification.

    The namespace comes from mlx-lm's own parser (see :func:`main`), so a new
    mlx-lm field gets its default instead of an AttributeError; this function
    only overwrites what the run actually decides.
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
        {
            "kind": "params",
            "pct": 100.0 * trainable / total,
            "detail": f"{trainable / 1e6:.3f}M/{total / 1e6:.3f}M",
        }
    )


def install_params_hook(lora_module: Any) -> None:
    """Route mlx-lm's trainable-parameter report into :func:`emit_params`.

    ``train_model`` reports those counts by printing a line; it reaches the
    printer through a module global, so replacing that global in this child -
    which is ours, single-threaded, and about to train - is how the counts become
    data. The lookup is checked first: if mlx-lm moves the printer, this fails
    loudly here instead of silently dropping the metric.
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
    """A callback that turns mlx-lm's reports into the parent's metric dicts."""

    class Emitter(training_callback):  # type: ignore[misc, valid-type]
        def on_train_loss_report(self, info: dict) -> None:
            emit(
                {
                    "kind": "train",
                    "iter": int(info["iteration"]),
                    "loss": float(info["train_loss"]),
                    "tok_s": int(info["tokens_per_second"]),
                }
            )

        def on_val_loss_report(self, info: dict) -> None:
            # mlx-lm reports the iteration *before* the evaluation it just ran
            # ("iteration": it - 1 in tuner/trainer.py), so +1 keeps both curves
            # on the same axis as the training reports.
            emit(
                {
                    "kind": "val",
                    "iter": int(info["iteration"]) + 1,
                    "loss": float(info["val_loss"]),
                }
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
