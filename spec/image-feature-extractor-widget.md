# Image Feature Extractor — an interpretable image-classification component for Orange

**Status:** idea / design sketch (no code yet)
**Context:** educational use
**Companion documents:** `nn-suite-architecture.md` (kernel-first layering, packaging, and build order — binding for all suite components); general discussion of image classification in Orange (Image Embedding / transfer learning vs. hand-crafted features)

---

## 1. Motivation

Modern image classification in Orange (Image Analytics add-on) works by transfer learning:
**Import Images → Image Embedding** (a pre-trained CNN such as Inception v3 or SqueezeNet
compresses each image to a ~2048-dim feature vector) → any standard learner (SVM, kNN,
Random Forest, Neural Network/MLP).

This is powerful but opaque: nobody can say *why* an embedding classifies a cloud as a cloud.
For educational purposes we want the opposite trade-off:

> Extract a **small set of human-understandable features** from each image — line probes,
> profiles, texture statistics, symmetry scores — and feed them to Orange's standard
> Neural Network widget. Every stage of the pipeline is then inspectable, and every
> classification can be explained in words.

Example explanations the approach should yield:

- "Classified as *cloud* because the row profile has two prominent peaks and the
  fine-texture score is high."
- "Classified as *8* because the hole count is 2 and stroke crossings are maximal near the
  image centre."

Target tasks (both binary and multi-class):

1. **Clouds vs. no clouds** (sky-facing imagery) — classes differ in gross intensity
   topology and texture. Realistic accuracy: **~90%+** with features below.
2. **Handwritten digit recognition** — classes differ in stroke topology and spatial
   mass distribution. Realistic accuracy: **~85–93%**, honestly stated against ~99% for a
   purpose-trained CNN. The gap, and why it exists, is itself a core lesson.

---

## 2. Evaluation of the approach

**Why it is sound**

- Small feature count (~35) means small data requirements: a few hundred labelled images
  suffice, and overfitting is controllable.
- Everything is auditable end-to-end: feature values in Data Table, distributions in
  Feature Statistics, feature–class relations in Scatter Plot/Box Plot, model in the
  Neural Network widget, evaluation in Test & Score and Confusion Matrix.
- Strong historical precedent: projection profiles, crossing counts and hole counts were
  the standard OCR toolkit before ~1990. The idea generalises those.

**Known weaknesses (design must address them)**

| Weakness | Mitigation |
|---|---|
| Peak counting is ill-posed at the "one peak with shoulder vs. two peaks" boundary (scale-space problem) | Keep continuous quantities (prominences, widths, positions) alongside the count; use prominence-based peak definitions |
| Sparse line probes are position-sensitive (a shifted cloud/digit moves the peaks) | Use *dense* projections (every row/column summarised) and require normalisation (centre/crop, contrast) first |
| Denoising can erase the signal (thin cirrus is a small perturbation) | Make the denoising scale an explicit, configurable parameter, chosen relative to the smallest feature scale of interest |
| A neural net on top of interpretable features re-introduces a black box | Compare NN against decision tree and logistic regression; if all agree within a couple of points, the features — not the model — carry the intelligence (itself a key lesson) |
| Hidden human overfitting (feature choices tuned by eyeballing the training set) | Evaluate on held-out data; test **stability** under small shifts/jitters and varying denoising settings |
| Trivial baselines can win (mean brightness alone may detect clouds) | Always report baselines: (a) trivial global statistics, (b) raw downsampled pixels → same learner, (c) Image Embedding → SVM |
| Absolute thresholds in features silently encode dataset lighting | Use adaptive/quantile-based definitions ("fraction above this image's own 75th percentile") |

**A deliberate trap, kept as a lesson:** with colour retained, "fraction of bluish pixels"
solves cloud detection trivially — a shortcut that teaches nothing about spatial structure.
Compute all features on **luminance only**; expose colour features as an optional,
switchable group to demonstrate the shortcut effect.

---

## 3. Widget design

A thin Orange widget shell, provisionally **Image Feature Extractor**. All feature
computation lives in the `nn-kernel` package (`nn_kernel.features`); the widget
adds options, preview, and Orange signal adapters only. See
`nn-suite-architecture.md` for the kernel-first layering.

### Inputs / outputs

- **Input:** Images (the table emitted by Import Images; image path carried as a meta
  attribute).
- **Output:** Features (a standard Orange Table: one row per image, ~35 numeric feature
  columns, class column inherited from the input, and the image path kept as a meta column
  so any suspicious row can be traced back to its source image — debuggability is the
  point of the whole exercise).

### Options panel

- Normalisation: centre/crop, contrast-normalise (on by default).
- Denoising: on/off + strength (scale parameter, explicit).
- Feature groups A–F with on/off toggles (group ablation as an experiment, see §5).
- Optional colour-feature group (default off).

### Main area (nice-to-have)

- Preview of the current image with the row/column projection profiles plotted alongside
  — makes "the feature" visible, not just a number.

### Engineering notes

- **All computation lives in the `nn_kernel.features` subpackage** (pure Python,
  Orange-free); the widget is a thin shell of options, preview, and signal adapters.
  Prototype and iterate in the kernel with pytest (feature stability under small
  affine jitter is a first-class test); widget boilerplate (`OWWidget`,
  inputs/outputs, `Setting` persistence, progress bar) comes last.
- **Scale features** inside the widget (or pipe through Preprocess → Normalize): the MLP
  is scale-sensitive and projection sums are ~1000× larger than skewness values.
- Single-file widget to start; package as a minimal add-on later.

---

## 4. Feature catalogue (~35 features, six groups)

All computed on a centred, contrast-normalised, denoised **luminance** image.
Each feature is a human-auditable sentence.

### A. Global photometry (6) — overall brightness / contrast character

| # | Feature | Interpretation |
|---|---|---|
| 1 | Mean intensity | Overall brightness |
| 2 | Intensity standard deviation | Contrast |
| 3 | Robust contrast (p90 − p10) | Contrast without outlier sensitivity |
| 4 | Skewness | Bright blobs on dark vs. dark on bright |
| 5 | Kurtosis | Flat vs. peaky intensity distribution |
| 6 | Bright fraction (above own median + robust contrast/2) | Adaptive "how much bright stuff" |

### B. Profile summaries (12) — generalised line probes, made *dense*

Build the full row-projection (mean intensity per row) and column-projection arrays,
then summarise each — coverage of every line, not six arbitrarily chosen ones.

| # | Feature | Interpretation |
|---|---|---|
| 7 | Row profile trend (linear slope) | Brighter at top or bottom? |
| 8 | Row profile roughness (total variation / mean) | Lumpy vs. smooth vertically |
| 9 | Row profile peak count (prominence-based) | Number of horizontal bands of "stuff" |
| 10 | Row profile centre of mass | Where the mass sits vertically |
| 11–14 | Same four for the **column profile** | Vertical analogue of 7–10 |
| 15 | Main-diagonal profile peak count | Diagonal band structure |
| 16 | Anti-diagonal profile peak count | Diagonal band structure (other way) |
| 17 | Row × column profile correlation | Do horizontal and vertical structures agree? (lumpy vs. streaky) |
| 18 | Median up-crossing count | How often profiles cross their median level — stable, parameter-free cousin of peak counting |

### C. Spatial mass distribution (6) — where the bright matter sits

| # | Feature | Interpretation |
|---|---|---|
| 19 | Intensity centre of mass, x | Left/right placement |
| 20 | Intensity centre of mass, y | Top/bottom placement |
| 21 | Horizontal spread | Wide vs. narrow bright mass |
| 22 | Vertical spread | Tall vs. squat |
| 23 | Elongation (second-moment eigenvalue ratio) | Streak vs. blob |
| 24 | Radial gradient | Mass concentrated at centre vs. edges |

### D. Texture / smoothness (6) — the clouds workhorse

| # | Feature | Interpretation |
|---|---|---|
| 25 | Mean gradient magnitude | Overall roughness |
| 26 | Edge density (adaptive threshold) | Fraction of "edgy" pixels |
| 27 | Fine texture (mean local std, 5×5) | Small-scale lumpiness |
| 28 | Coarse texture (variance of 16×16 block means) | Big lumps vs. uniform |
| 29 | Coarse-to-fine texture ratio | Puffy cumulus vs. wispy cirrus live at different scales |
| 30 | Autocorrelation half-width | Typical feature size in pixels |

### E. Topology (3) — the classic digit killers

| # | Feature | Interpretation |
|---|---|---|
| 31 | Hole count (Euler number) | "8"=2, "0"=1, "1"=0 — the most interpretable digit feature ever |
| 32 | Foreground area fraction | Ink / cloud coverage |
| 33 | Stroke crossing counts (mean crossings of median level per row) | Dense version of the classic OCR feature; open strokes ("1", "7") vs. closed ("8", "0") |

### F. Symmetry (3) — cheap, powerful, very visual

| # | Feature | Interpretation |
|---|---|---|
| 34 | Left-right symmetry (mirrored-half correlation) | Bilaterally symmetric structure |
| 35 | Top-bottom symmetry | Vertical mirror |
| 36 | Quadrant self-similarity (mean inter-quadrant correlation) | Uniform vs. localised structure |

**Expected group–task affinity:** clouds should live in groups **D + B**;
digits in groups **E + C + B**. (Verify by ablation — see §5.)

---

## 5. Workflow and experiments

### Canvas workflow

```
Import Images → Image Feature Extractor → Data Table / Feature Statistics
                                         → Neural Network → Test & Score → Confusion Matrix
                                         → (comparison: Tree, Logistic Regression)
```

### Evaluation protocol

1. Stratified cross-validation; report confusion matrix; examine errors by hand
   ("called this overcast sky *clear* because the profile is flat but elevated" — the
   diagnostic sentence the whole design aims to produce).
2. Baselines to beat: (a) trivial global statistics, (b) raw downsampled pixels → same
   learner, (c) Image Embedding → SVM as the modern reference.
3. Stability tests: small affine jitter and varying denoising strength must not flip
   feature categories (e.g. "two peaks" → "one peak"); if they do, the feature is noise.

### Two experiments that frame the lesson

1. **Ablation** — toggle feature groups and watch accuracy move in Test & Score.
   If group A alone classifies clouds (mean brightness doing all the work), that is a
   finding worth discussing, not a failure.
2. **Model-transparency check** — NN vs. decision tree vs. logistic regression on
   identical features. If all land within a couple of points, the *features* carried the
   intelligence and the neural net is mostly ceremony — a profound point about where
   representation lives. If the NN clearly pulls ahead, inspect which nonlinearity it
   found.

---

## 6. Roadmap

1. Small labelled image folders (clouds/not-clouds; digits).
2. `nn_kernel.features` prototype: groups A–F with pytest coverage; iterate on
   feature definitions and stability tests.
3. `Image Feature Extractor` widget shell over the kernel; add profile-preview
   main area and group toggles.
4. Run the evaluation protocol and both experiments; record findings.
5. Optional: compare against `orange3-imagenets` (experimental add-on that visually builds
   and trains small Keras CNNs inside Orange) to contrast hand-crafted features with a
   trained-from-scratch CNN in the same canvas.

See `nn-suite-architecture.md` for the kernel-first layering and the cross-suite
build order.