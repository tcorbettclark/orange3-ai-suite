# NN Suite Architecture — binding document

**Status:** design doc (no code yet)
**Context:** binds the two domain designs into one build plan and records all
cross-cutting decisions
**Companion documents:** `image-feature-extractor-widget.md` (image domain),
`tiny-lm-suite-design.md` (language-model domain). Domain-specific design detail
lives in those; this document owns the shared architecture, packaging, testing
strategy, and build order.

---

## 1. Purpose

The suite consists of six Orange widgets (one existing, five new) built around a
shared pure-Python kernel. The kernel serves two teaching streams simultaneously:
the Orange stream (widgets in the canvas) and a future Python-only stream (the same
library used headless). The widgets are thin shells; the substance is the kernel.

---

## 2. Layered architecture

```
┌────────────────────────────────────────────────────────────┐
│ orange3-nn-suite (single Orange add-on)                   │
│   widgets: options panels · live plots · signal adapters   │
│   imports Orange types (Table, Corpus) and the kernel      │
├────────────────────────────────────────────────────────────┤
│ nn-kernel (pure Python package; no Orange, no Qt, ever)    │
│   nn_kernel/features/   image feature groups A–F           │
│   nn_kernel/lm/         tokenizer, windowing, Keras        │
│                         builders, sampling, model objects  │
│   nn_kernel/lora/       MLX QLoRA wrappers                 │
│   speaks: NumPy arrays, strings, paths, its own objects    │
└────────────────────────────────────────────────────────────┘
```

**Disciplines (non-negotiable):**

1. The kernel never imports Orange or Qt. All Orange-type conversion (Table,
   Corpus, meta columns) lives in the widget layer as thin adapters.
2. Model-object conventions — the `generate(prompt, temperature)` interface,
   tokenizer-travels-with-model, adapter-serialized-with-base-model — are defined
   in the kernel, so Keras and MLX models interoperate through kernel types.
3. Widgets declare a pinned kernel dependency (`nn-kernel>=x,<y`); the
   `mlx-lm` version pin lives inside the kernel's lora extra.
4. The kernel API is treated as internal-until-stable; no premature generality —
   it is a course kernel with three subpackages, not a framework.

---

## 3. Packaging

### `nn-kernel` — the substance

Pure Python package, pip-installable, with extras so students install only what
each course stage needs:

| Extra | Brings | Used by |
|---|---|---|
| `[image]` | OpenCV | Image Feature Extractor |
| `[lm]` | Keras (+ tf_kera pin) | LM Trainer, LM Generator, Vocab Inspector |
| `[lora]` | mlx-lm`[train]` (Apple-Silicon-only, optional) | LoRA Trainer |

### `orange3-nn-suite` — the shells

A **single Orange add-on**, toolbox category **"Neural Networks"**, containing all
six widgets. `orange3-nnviewer` (existing, featherweight, dependency-light) is
**folded in**: its viewer becomes part of the suite, and its `--no-deps`
thin-shell discipline remains the reference pattern for every widget here. One
install, one category, one story for students.

**License: MIT** for all suite code (`REPO-CONTEXT.md` §3 records this and the related
caution: the community add-on that inspired the widget pattern is CC-BY-NC-SA —
inspiration only, never code copying).

| Widget | Kernel behind it | Extra needed |
|---|---|---|
| NN Viewer *(existing, folded in)* | — (reads scikit-learn MLP model objects directly) | none |
| Image Feature Extractor | `nn_kernel.features` | `[image]` |
| LM Trainer | `nn_kernel.lm` | `[lm]` |
| LM Generator | `nn_kernel.lm` / `nn_kernel.lora` model objects | `[lm]` and/or `[lora]` |
| LoRA Trainer | `nn_kernel.lora` | `[lora]` |
| Vocab Inspector | tokenizer objects from `nn_kernel.lm` | `[lm]` |

Widgets degrade gracefully when an extra is missing (clear "install
nn-kernel[lm]" style message), matching the MLX-on-non-Apple policy already agreed.

---

## 4. Shared design principles (cross-suite)

- **State travels inside objects** between widgets (tokenizer embedded in model;
  adapter serialized with its base model) — mismatch bugs unrepresentable.
- **Live training feedback** via the established QThread + callback → pyqtgraph
  pattern, consuming loss/perplexity values the kernel emits.
- **Read-only inspectors never enter the pipeline** (Vocab Inspector has no
  outputs; NN Viewer takes a model and draws it).
- **No new evaluation or visualization components** — Test & Score, Confusion
  Matrix, Orange3-Text, and the evaluation spine are all existing Orange widgets.
- **Documented non-choices live with their domain** (see §11 of
  `tiny-lm-suite-design.md` and the trade-off tables of
  `image-feature-extractor-widget.md`); cross-cutting ones live here (below).

## 5. Cross-cutting non-choices

| Not chosen | Why |
|---|---|
| One merged design document | Domain docs stay self-contained and reviewable; this binding doc owns shared decisions once |
| Separate add-on per domain | Single add-on guarantees one "Neural Networks" category and one install; dependency hygiene achieved via kernel extras instead |
| Kernel logic inside widgets (the original two-doc plan) | ~80% of the substance is Orange-free; Orange's bundled-Python environment is fragile; the Python-only stream reuses the same code |
| Widget-layer model conventions | Conventions defined at the widget layer would not survive the Python stream; the kernel owns them |
| `orange3-nnviewer` left standalone | Folding it in gives the course one install and one category; its minimal-shell pattern becomes the suite's discipline |

---

## 6. Testing strategy

- **Kernel:** pytest, no Qt, no Orange. Highest-value targets: tokenizer edge cases
  (BOS/padding, sentence boundaries), window construction, sampling determinism
  under fixed seeds, feature stability under small affine jitter (a feature whose
  category flips on a 5-pixel shift is noise), LoRA artifact round-trip.
- **Widgets:** offscreen-Qt tests (the nnviewer precedent: `pytest -q` with Qt
  rendering offscreen, no display needed).
- **Adapters:** signal-plumbing tests against real Orange data types — the one
  place where Orange objects legitimately appear in tests.

---

## 7. Build order

1. `nn-kernel` skeleton; `nn_kernel.features` prototype (the image-feature
   experiments live here from day one) + pytest coverage.
2. `nn_kernel.lm`: tokenizer first (Vocab Inspector logic early — it debugs the
   tokenizer), then windowing, bigram/MLP-window/LSTM builders, sampling loop.
3. `orange3-nn-suite` shells + adapters; fold in NN Viewer; Image Feature
   Extractor, LM Trainer, LM Generator, Vocab Inspector widgets.
4. `nn_kernel.lora` last, behind the BritLLM/MLX action items; LoRA Trainer shell.
5. End-to-end runs of both arcs (image classification; three-stage LLM arc);
   record findings; tune timescales for the 16 GB Mac Mini classroom.

The planned Python Script widget prototypes from the domain docs are *replaced
by* kernel development: same code, hosted in the kernel package from day one, tested
with pytest instead of inspected in the canvas.

---

## 8. Action items (pre-implementation, unchanged in substance)

1. Verify/produce an MLX conversion of BritLLM (`britllm/britllm-3b-v0.1`;
   Llama-family ⇒ expected to be supported; dry run required — assume no
   community conversion exists until checked).
2. Dry-run LoRA training time on a 16 GB Mac Mini (expect tens of minutes, not
   hours, on a small corpus).
3. Pin `mlx-lm[train]` inside the kernel's `[lora]` extra.
4. Confirm the toolbox shows one merged "Neural Networks" category for the
   combined add-on (expected trivially true for a single add-on; recorded so it
   is verified once, not assumed).