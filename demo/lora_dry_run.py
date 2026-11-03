"""LoRA timing dry run (action item §8.2): 4-bit QLoRA on the fetched model.

Times a small LoRA run with mlx_lm on `data/models/britllm-4bit` over a tiny
self-contained corpus, and reports wall-clock, iterations/second and a quick
adaptor-loaded generation check — the numbers the §8.2 record needs,
annotated with the machine profile (our dev machine vs. the classroom's
16 GB Mini).

Hyperparameters mirror the reference LoRA runs (300 iters, lr 1e-4,
16 layers, batch 4, seq len 512) — tuned elsewhere, reused here as the
starting defaults for nn_kernel.lora.

    uv run python demo/lora_dry_run.py
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx  # noqa: F401  (mlx presence check)

from mlx_lm import generate, load

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODEL = DATA / "models" / "britllm-4bit"
CORPUS_DIR = DATA / "dryrun"
ADAPTER = DATA / "dryrun-adapter"

ITERS = 100  # a few minutes budget; the reference used 300 (background)

# A tiny public-domain-flavoured completion corpus, enough sentences for
# batch 4 x seq 512 x ~100 iters. Content is placeholder "domain" text —
# the *timing* is what this run records, not the language lesson.
SAMPLE_TEXTS = [
    "The king gathered his counsellors in the great hall, and spoke of war. "
    "The north wind carried tidings from the coast, and the old men nodded gravely.",
    "Beneath the spreading oak, the minstrel sang of days long past, when banners flew "
    "and horses thundered across the fields of the shire at first light.",
    "The physician arrived at dusk, his satchel heavy with herbs. He bled the fevered "
    "patient and prescribed rest, broth, and a tincture of willow bark thrice daily.",
    "To every season there is an appointed time. The tide obeys the moon, the blossom "
    "obeys the spring, and the patient scribe records what heaven decrees.",
    "The merchant counted his coins by candlelight, muttering of tariffs, taxes and "
    "the dishonesty of strangers from the eastern ports who sold pepper at half price.",
    "Rain fell upon the thatched roofs of the village for seven days, and the river "
    "rose until the miller could no longer grind. The villagers prayed, and waited.",
    "In the name of the crown I command thee: deliver the writ to the sheriff before "
    "the third bell, lest the assizes fail and the county be fined two hundred marks.",
    "She mixed flour, salt and ale into dough, let it rise by the hearth, then baked "
    "the loaf in the stone oven until the crust sang when tapped.",
]

EVAL_SENTINEL = "The old scribe wrote at dawn"


def make_corpus() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, n_rows, seed in (("train.jsonl", 120, 0), ("valid.jsonl", 12, 1)):
        path = CORPUS_DIR / name
        if path.exists():
            return  # idempotent
        path.write_text("\n".join(
            json.dumps({"text": SAMPLE_TEXTS[(i + seed * 3) % len(SAMPLE_TEXTS)]})
            for i in range(n_rows)
        ) + "\n")


def train() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora",
         "--model", str(MODEL),
         "--train",
         "--data", str(CORPUS_DIR),
         "--adapter-path", str(ADAPTER),
         "--iters", str(ITERS),
         "--learning-rate", "1e-4",
         "--num-layers", "16",
         "--batch-size", "4",
         "--max-seq-length", "512",
         "--steps-per-report", "20",
         "--steps-per-eval", "50",
         "--seed", "42"],
        capture_output=True, text=True,
    )
    log = result.stdout + result.stderr
    if result.returncode != 0:
        raise SystemExit(f"mlx_lm lora failed:\n{log[-2000:]}")
    (ADAPTER / "run.log").write_text(log, encoding="utf-8")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log)  # strip rich/ANSI colour
    params = re.search(
        r"Trainable parameters:\s*([\d.]+)%\s*\(([\d,.]+M?)\s*/\s*([\d,.]+M?)\)", clean
    )
    tok_rates = [
        int(m) for m in re.findall(r"\d+\s+[\d.]+\s*\u25bc\s+(\d+)\s+\d", clean)
    ]
    return {
        "trainable_params_pct": params.group(1) if params else None,
        "trainable_params": f"{params.group(2)} / {params.group(3)}" if params else None,
        "tok_s_median": (sorted(tok_rates)[len(tok_rates) // 2]
                         if tok_rates else None),
    }


def main() -> None:
    if not (MODEL / "model.safetensors").exists():
        raise SystemExit("run demo/fetch_britllm.py first")
    make_corpus()
    t0 = time.perf_counter()
    stats = train()
    wall = time.perf_counter() - t0

    model, tokenizer = load(str(MODEL), adapter_path=str(ADAPTER))
    sample = generate(model, tokenizer, prompt=EVAL_SENTINEL, max_tokens=24).strip()

    report = {
        "machine": {
            "chip": platform.machine(),
            "python": sys.version.split()[0],
            "mac": "Apple Silicon (this dev machine — annotate vs. 16 GB Mini classroom)",
        },
        "model": {"path": str(MODEL), "quant": "4-bit affine, group 64, ~2.67 B params"},
        "training": {"iters": ITERS,
                     "tok_s_median": stats["tok_s_median"],
                     "wall_seconds": round(wall, 1),
                     "trainable_params": stats["trainable_params"],
                     "trainable_params_pct": stats["trainable_params_pct"]},
        "generation": {EVAL_SENTINEL: sample},
    }
    print(json.dumps(report, indent=1))
    (ADAPTER / "dry-run-report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()