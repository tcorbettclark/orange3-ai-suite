# demo/

Self-contained teaching workflows for the two image tasks of the suite.

## Contents

| File | Role |
|---|---|
| `fetch_mnist.py` | Regenerates `mnist/` (idempotently; needs the dev venv `[lm]` for keras) |
| `mnist/` | **Committed**: 6,000 PNGs — MNIST digits upscaled to 56×56, seeded stratified subset (500/class train, 100/class val), one subfolder per label (Import Images layout) |
| `fetch_clouds.py` | Curated cloud-corpus fetcher (see below); network needed, ~5 min |
| `fetch_britllm.py` | Fetches `britllm/britllm-3b-v0.1` (cache-aware) and converts it to MLX 4-bit → `data/models/britllm-4bit` (§8.1) |
| `lora_dry_run.py` | Times a small 4-bit LoRA run on the converted model + adapter-loaded generation check (§8.2; report in `data/dryrun-adapter/`) |
| `mnist_features.ows` / `cloud_features.ows` | **Committed** canvas workflows (generated via `make_workflows.py`; Import Images paths inside are machine-local — regenerate or re-point the File widget if they don't match) |

## Stage-2/3 arc record (run_stages23.py; `--effort low` ≈ 2.5 min,
`medium`/`high` as the longer records; report in data/stage23-report.json,
generation transcript in data/review/stage23_transcript.txt)

Findings at teaching effort (--effort low, 30 CPT + 20 SFT iters):
- the raw base model is fluent and — for a *famous* domain — already answers
  domain facts (Gibbon is in web pre-training); "raw model" ≠ ignorant;
- the Stage-3b SFT beat shows behaviourally: `A:` prompts elicit
  assistant-pattern completions and corpus-dominant phrases, WITH degenerate
  repetition + <unk> runs at 20 iters (instructive honest artefact);
- perplexity nuance: modern boilerplate scores ppl 2.6 vs Gibbon 13.3 on the
  base model — perplexity tracks generic predictability, not domain depth;
  held-out ppl worsened after 20–30 QLoRA iters (perturbation, not
  memorisation) — the perplexity *drop* is a high-effort finding.
- britllm's tokenizer ships no chat template (raw base model); our converted
  copy gets a minimal generic one at conversion time (demo/fetch_britllm.py)
  because mlx-lm's prompt-masking SFT path requires it;
- kernel: LoraData.from_pairs/from_texts now validate valid-vs-batch-size.

## Stage-1 measured timescales (run_stage1.py, dev M4/32 GB; a classroom Mini
is ~1.5–3× slower)

| Tier | Time | Result |
|---|---|---|
| bigram (k=1, e5) | 3.2 s | loss 2.84 → 2.36 — the plateau |
| MLP-over-window (k=16, emb 32, hidden 128, e8, 79 k params) | **1.7 s/epoch** (14 s) | loss 2.21 → 1.50, val 1.56 |
| LSTM (k=32, emb 64, units 128, 1 layer, e2) | 50.7 s/epoch | val 1.67 in 2 epochs (≈1.6 within e5 extrapolated) → good results ≈ 1–9 min |

Sample evolution (greedy, from `the `): e1 `the sustine of the seland the
empereat of the semplest` → e3 `the and of the emperor and the emperor of
the senate` → e7 `the arms of the senate was and of the senate` — the
gibberish → domain-pastiche lesson in three epochs, seconds per epoch. The
spec's "minutes" budget is generous; classroom runs can afford deeper
models or more epochs.

## The workflows

```
Import Images → Image Feature Extractor → Test & Score ← Neural Network
                     ↓↓
        Data Table, Column Statistics
                                        Test & Score → Confusion Matrix
```

Open in the Orange canvas with this add-on installed
(`/Applications/Orange.app/Contents/MacOS/python -m Orange.canvas`, then
File → Open → `demo/*.ows`). The Image Feature Extractor's settings
(working size; feature groups A–F) are preset and editable in the widget:
group toggles are the ablation experiment; watch the confusion matrix change.

MNIST: working size 56 (why: 28-wide images starve the 5×5/16×16 texture
features). Clouds: 128.

## Measured results (the honest ceiling on this corpus)

5-fold CV, Orange's Neural Network learner (the canvas default path)
reproduced headlessly by `demo/diagnose_cloud.py` / `demo/probe_colour.py`
(kept as the record of these measurements):

| Feature set | CA |
|---|---|
| 36 luminance features (A–F), raw | ~0.37 |
| 36 z-scored | ~0.35–0.40 |
| + colour group G (the trap) | ~0.38–0.44 |
| majority baseline | 0.225 |

Interpretation: genera genuinely confound under hand-crafted features on this
kind of photography (Atlas catalogue photos are often close-ups naming two
genera in their titles, or twilight shots); colour is the single biggest
lever — which is exactly why it is the spec's designated trap group. The
confusion matrix *is* the lesson: read which genera confound and
ask which spatial structure the missing feature would capture.

## Cloud corpus provenance (fetch_clouds.py)

Images come from the **WMO International Cloud Atlas** (WMO-No. 407,
cloudatlas.wmo.int) — the World Meteorological Organization's definitive
reference standard for cloud classification, at genus level:

- `ci` Cirrus · `ac` Altocumulus · `sc` Stratocumulus · `cb` Cumulonimbus · `cu` Cumulus
- every photograph is a ground-observation photograph (the Atlas' own
  filename tags distinguish them from sketches/satellite/aerial views, and
  the fetcher filters on the genus itself);
- fetched to `data/clouds/` — **untracked**: personal, non-commercial use,
  exactly per the Atlas copyright terms. The images remain © their
  photographers (credited per image — photographer, location, date, Atlas
  image ID — in `data/clouds/CREDITS.md`) and are **not** covered by this
  repository's MIT code licence;
- contact sheets for a quick eyeball pass land in `data/review/`.

MNIST digits are widely redistributed (see e.g. the MNIST database webpage);
keras.datasets.mnist is the acquisition channel.

## Regeneration

```bash
uv run python demo/fetch_mnist.py       # rebuilds mnist/ subset deterministically
uv run python demo/fetch_clouds.py      # refetches/refreshes data/clouds + credits
uv run python demo/make_workflows.py    # rebuilds both .ows (canvas API)
```