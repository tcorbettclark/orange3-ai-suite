"""Run the Stage-1 arc at teaching scale on the Gibbon corpus (build order §7
step 5 for the LM stream): the architecture ladder, live, with measured
timescales — the record that tunes the classroom plan.

    uv run python demo/run_stage1.py [--epochs MLP_EPOCHS] [--skip-lstm]

Writes data/stage1-report.json + data/review/stage1_samples.txt (the
gibberish -> pastiche evolution).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.neural_network")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from nn_kernel.lm import Tokenizer, train_language_model

BASE = Path(__file__).resolve().parent
CORPUS = BASE / "gibbon_corpus.txt"
REPORT = BASE.parent / "data" / "stage1-report.json"
SAMPLES_TEXT = BASE.parent / "data" / "review" / "stage1_samples.txt"


def run_tier(name, texts, tokenizer, *, context_size, epochs, on_epoch, seed=0, **kw):
    t0 = time.perf_counter()
    model = train_language_model(
        texts, tokenizer, name,
        context_size=context_size, epochs=epochs, seed=seed,
        on_epoch=on_epoch, **kw,
    )
    wall = time.perf_counter() - t0
    return wall, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--only-lstm", action="store_true")
    args = parser.parse_args()

    paragraphs = [
        p for p in CORPUS.read_text(encoding="utf-8").split("\n\n") if p.strip()
    ]
    print(f"corpus: {len(paragraphs)} paragraphs")
    char_tok = Tokenizer.from_corpus(paragraphs, "char")
    samples: list[str] = []
    report = {"machine": platform.platform(), "char_vocab": char_tok.vocab_size}

    # ---- bigram (tier 0): the floor, seconds ----
    if not args.only_lstm:
        def bigram_hook(stats):
            samples.append(f"bigram e{stats['epoch']}: {stats['sample']!r}")

        wall, bigram = run_tier(
            "bigram", paragraphs, char_tok, context_size=1, epochs=5,
            on_epoch=bigram_hook, sample_prompt="", sample_temperature=0.0,
            sample_max_new_tokens=40,
        )
        report["bigram"] = {"wall_s": round(wall, 1),
                            "final_loss": bigram.hyperparameters["final_loss"]}
        print(f"bigram: {wall:.1f}s, loss -> {bigram.hyperparameters['final_loss']:.3f}")

    # ---- MLP-over-window (tier 1, flagship) ----
    mlp_samples: list[str] = []

    def mlp_hook(stats):
        mlp_samples.append((stats["epoch"], stats["loss"], stats["val_loss"],
                            stats.get("sample", "")))

    if not args.only_lstm:
        wall, mlp = run_tier(
            "mlp_window", paragraphs, char_tok, context_size=16,
            embedding_dim=32, hidden_units=128,
            epochs=args.epochs, batch_size=256, seed=0,
            on_epoch=mlp_hook,
            sample_prompt="the ", sample_temperature=0.0,
            sample_max_new_tokens=60,
        )
        report["mlp_window"] = {
            "wall_s": round(wall, 1), "epochs": args.epochs,
            "final_loss": round(mlp.hyperparameters["final_loss"], 3),
            "final_val_loss": round(mlp.hyperparameters["final_val_loss"], 3),
            "per_epoch_s": round(wall / args.epochs, 1),
            "**parameters**": mlp.parameter_count(),
        }
        print(f"mlp_window: {wall:.1f}s ({wall/args.epochs:.1f}s/epoch), "
              f"loss {mlp.hyperparameters['final_loss']:.3f}, "
              f"val {mlp.hyperparameters['final_val_loss']:.3f}, "
              f"{mlp.parameter_count()} params")

    # ---- LSTM (tier 2): launch-and-return; measure the step rate ----
    if not args.skip_lstm:
        lstm_samples = []

        def lstm_hook(stats):
            lstm_samples.append((stats["epoch"], stats["loss"],
                                 stats.get("sample", "")))

        wall, lstm = run_tier(
            "lstm", paragraphs, char_tok, context_size=32,
            embedding_dim=64, lstm_units=128, lstm_layers=1,
            epochs=2, batch_size=128, seed=0,
            on_epoch=lstm_hook, sample_prompt="the ",
            sample_temperature=0.0, sample_max_new_tokens=60,
        )
        report["lstm"] = {
            "wall_s": round(wall, 1), "epochs": 2,
            "per_epoch_s": round(wall / 2, 1),
            "**parameters**": lstm.parameter_count(),
            "final_val_loss": round(lstm.hyperparameters["final_val_loss"], 3),
        }
        print(f"lstm: {wall:.1f}s ({wall/2:.1f}s/epoch), "
              f"val {lstm.hyperparameters['final_val_loss']:.3f}")
        samples.extend(f"lstm e{e} (loss {l:.3f}): {s!r}"
                       for e, l, s in lstm_samples if s)

        samples.extend(
            f"mlp e{e} (loss {l:.3f}, val {v:.3f}): {s!r}"
            for e, l, v, s in mlp_samples if s
        )
    SAMPLES_TEXT.write_text("\n\n".join(samples), encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"\nreport: {REPORT}\nsamples: {SAMPLES_TEXT}")
    for line in samples[-8:]:
        print(" ", line)


if __name__ == "__main__":
    main()