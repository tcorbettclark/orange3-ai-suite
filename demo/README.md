# demo/

Self-contained teaching workflows for the two image tasks of the suite.

## Contents

| File | Role |
|---|---|
| `fetch_mnist.py` | Regenerates `mnist/` (idempotently; needs the dev venv `[lm]` for keras) |
| `mnist/` | **Committed**: 6,000 PNGs — MNIST digits upscaled to 56×56, seeded stratified subset (500/class train, 100/class val), one subfolder per label (Import Images layout) |
| `fetch_clouds.py` | Curated cloud-corpus fetcher (see below); network needed, ~5 min |
| `mnist_features.ows` / `cloud_features.ows` | **Committed** canvas workflows (generated via `make_workflows.py`; Import Images paths inside are machine-local — regenerate or re-point the File widget if they don't match) |

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