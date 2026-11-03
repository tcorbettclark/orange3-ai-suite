"""Run the Stage-2 / Stage-3 record on the Gibbon corpus (build order §7
step 5, LLM stream): the full three-stage arc the spec teaches, measured.

Stage 2 — the raw base model (adapter off): fluent English, shallow in the
domain, poor instruction-following; in/out-of-domain perplexity.
Stage 3a — LoRA continued pre-training on the Gibbon paragraphs (the
{'text'} format, self-supervised): domain depth grows.
Stage 3b — LoRA SFT on hand-authored prompt–response pairs (the
{'prompt','completion'} format, prompt-masked loss): instruction-following
appears.

Every stage reports: greedy/deterministic generations on fixed prompts
(fluency + domain + instruction) and perplexity on a held-out Gibbon slice
vs. an embedded out-of-domain (modern English) block. Report:
data/stage23-report.json; transcript: data/review/stage23_transcript.txt

    uv run python demo/run_stages23.py [--skip-sft]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from nn_kernel.lora import LoraData, LoraOptions, LoRAModel, train_lora

BASE = Path(__file__).resolve().parent
MODEL = BASE.parent / "data" / "models" / "britllm-4bit"
CORPUS = BASE / "gibbon_corpus.txt"
REPORT = BASE.parent / "data" / "stage23-report.json"
TRANSCRIPT = BASE.parent / "data" / "review" / "stage23_transcript.txt"

# fixed prompts, chosen to expose the spec's three teaching questions:
PROMPTS = {
    "domain continuation": (
        "In the second century of the Christian era, the empire of Rome"
    ),
    "domain continuation 2": ("The legions of the Rhine were"),
    "instruction (unanswerable by a raw base model)": (
        "Q: Who wrote The History of the Decline and Fall of the Roman Empire?\nA:"
    ),
}
OUT_OF_DOMAIN = (
    "Welcome to the developer documentation for the mobile application. "
    "This release adds support for push notifications, offline caching, and "
    "a redesigned settings screen. Users should update the app through their "
    "device's store, then sign in with their account to sync preferences. "
    "Reported issues in the previous build, including crashes when editing "
    "large photo libraries and sync conflicts on shared devices, have been "
    "resolved. The engineering team continues to monitor crash reports and "
    "performance metrics, and the next release will focus on battery usage, "
    "startup time, and accessibility improvements across all supported "
    "platforms. Feedback from beta testers shaped several of these changes, "
    "and the changelog records each fix in detail. " * 6
)


def transcript_write(text: str, stage: str, block: list[str]) -> str:
    lines = [f"=== {stage} ===", ""]
    for name, prompt in PROMPTS.items():
        lines.append(f"[{name}]")
        lines.append(f"  prompt: {prompt!r}")
        lines.append(f"  output: {text.get((stage, name), '')!r}")
        lines.append("")
    lines.append("---")
    return "\n".join(lines)


# effort profiles: the arc is a *record of trends*, not a convergence claim;
# low keeps the whole three-stage loop inside ~1 minute.
EFFORTS = {
    "low":    {"cpt_iters": 30,  "sft_iters": 20,  "valid": 0.02,
               "steps_per_eval": 15, "steps_per_report": 10,
               "ppl_in": 1024, "ppl_out": 768, "gen_tokens": 32},
    "medium": {"cpt_iters": 100, "sft_iters": 50,  "valid": 0.02,
               "steps_per_eval": 25, "steps_per_report": 10,
               "ppl_in": 2048, "ppl_out": 1024, "gen_tokens": 48},
    "high":   {"cpt_iters": 300, "sft_iters": 100, "valid": 0.05,
               "steps_per_eval": 50, "steps_per_report": 20,
               "ppl_in": 4096, "ppl_out": 2048, "gen_tokens": 64},
}


def main() -> None:
    # (module-level EFFORTS above); argparse picks the profile
    parser = argparse.ArgumentParser()
    parser.add_argument("--effort", choices=EFFORTS, default="low",
                        help="low ~1 min, medium ~4 min, high = the full record")
    parser.add_argument("--skip-sft", action="store_true")
    args = parser.parse_args()
    effort = EFFORTS[args.effort]
    report: dict = {"effort": args.effort, "out_of_domain_chars": len(OUT_OF_DOMAIN)}
    paragraphs = [
        p for p in CORPUS.read_text(encoding="utf-8").split("\n\n") if p.strip()
    ]
    eval_text = "\n".join(paragraphs[-6:])  # held-out slice, in-domain
    max_gen = effort["gen_tokens"]
    generations: dict[tuple[str, str], str] = {}
    report: dict = {"out_of_domain_chars": len(OUT_OF_DOMAIN)}

    def run_generation_stage(label: str, adapter_path: str | None):
        model = LoRAModel(str(MODEL), adapter_path)
        model.adapter_enabled = adapter_path is not None
        block = []
        for name, prompt in PROMPTS.items():
            text = model.generate(prompt, temperature=0, max_new_tokens=max_gen,
                                  seed=0)
            generations[(label, name)] = text
            block.append(f"[{name}]\n  prompt: {prompt!r}\n  output: {text!r}\n")
        ppl_in = model.perplexity(eval_text, max_tokens=effort["ppl_in"])
        ppl_out = model.perplexity(OUT_OF_DOMAIN, max_tokens=effort["ppl_out"])
        report[label] = {
            "perplexity_in_domain": round(ppl_in["perplexity"], 2),
            "perplexity_out_domain": round(ppl_out["perplexity"], 2),
        }
        print(f"{label}: ppl_in={ppl_in['perplexity']:.2f} "
              f"ppl_out={ppl_out['perplexity']:.2f}")
        return "\n".join(block)

    if not (MODEL / "model.safetensors").exists():
        raise SystemExit("run demo/fetch_britllm.py first")

    stage2_block = run_generation_stage("stage2_base_model", None)

    # ---- Stage 3a: continued pre-training on Gibbon ----
    data = LoraData.from_texts(paragraphs[:-6], valid_fraction=effort["valid"])
    t0 = time.perf_counter()
    adapter = train_lora(
        MODEL, data, BASE.parent / "data" / "stage23-cpt-adapter",
        options=LoraOptions(
            iters=effort["cpt_iters"], batch_size=4, max_seq_length=512,
            num_layers=16, learning_rate=1e-4, seed=42,
            steps_per_eval=effort["steps_per_eval"],
            steps_per_report=effort["steps_per_report"],
        ),
        data_dir=BASE.parent / "data" / "stage23-cpt-data",
    )
    cpt_wall = time.perf_counter() - t0
    report["stage3_cpt"] = {"wall_s": round(cpt_wall, 1),
                            "iters": effort["cpt_iters"]}
    stage3_block = run_generation_stage("stage3a_continued_pretraining",
                                        adapter.adapter_path)

    blocks = [
        ("Stage 2 — raw base model (adapter off)", stage2_block),
        ("Stage 3a — + LoRA continued pre-training on Gibbon", stage3_block),
    ]

    # ---- Stage 3b: SFT on authored prompt–response pairs ----
    if not args.skip_sft:
        qa_pairs = [
            ("Q: In which century did the empire of Rome reach its greatest extent?\nA:",
             "in the second century of the Christian era."),
            ("Q: Who were the emperors named by Gibbon in the age of the Antonines?\nA:",
             "Nerva, Trajan, Hadrian, and the two Antonines."),
            ("Q: What guarded the extensive monarchy of Rome?\nA:",
             "ancient renown and disciplined valor."),
            ("Q: What influence cemented the union of the provinces?\nA:",
             "the gentle but powerful influence of laws and manners."),
            ("Q: What did the frontiers of the empire enjoy?\nA:",
             "the advantages of wealth and luxury."),
            ("Q: Who acquired the principal conquests of the Romans?\nA:",
             "the Romans achieved them under the republic."),
            ("Q: Which city was the seat of government in Britain?\nA:", "York."),
            ("Q: What did Gaul boast of under the empire?\nA:", "her twelve hundred cities."),
            ("Q: What was the condition of the imperial constitution?\nA:",
             "the image of a free constitution was preserved with decent reverence."),
            ("Q: What happened after the death of Marcus Antoninus?\nA:",
             "the empire entered its decline and fall."),
        ] * 6  # 60 pairs: the valid split must hold >= batch_size rows (mlx-lm's)
        data_sft = LoraData.from_pairs(
            [(q, c) for q, c in qa_pairs],
            # small pair corpora need a percentage that clears batch_size
            valid_fraction=max(0.1, effort["valid"]),
        )
        t0 = time.perf_counter()
        adapter_sft = train_lora(
            MODEL, data_sft, BASE.parent / "data" / "stage23-sft-adapter",
            options=LoraOptions(
                iters=effort["sft_iters"], batch_size=4, max_seq_length=128,
                num_layers=16, learning_rate=1e-4, seed=42,
                steps_per_eval=effort["steps_per_eval"],
                steps_per_report=effort["steps_per_report"],
            ),
            data_dir=BASE.parent / "data" / "stage23-sft-data",
        )
        sft_wall = time.perf_counter() - t0
        report["stage3_sft"] = {"wall_s": round(sft_wall, 1),
                                "iters": effort["sft_iters"]}
        sft_block = run_generation_stage("stage3b_sft_pairs", adapter_sft.adapter_path)
        blocks.append(("Stage 3b — + LoRA SFT on authored pairs (prompt-masked)",
                       sft_block))

    transcript = "\n\n".join(title + "\n" + body for title, body in blocks)
    TRANSCRIPT.write_text(transcript, encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"\nreport: {REPORT}\ntranscript: {TRANSCRIPT}")


if __name__ == "__main__":
    main()