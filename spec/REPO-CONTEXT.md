# REPO-CONTEXT — entry point for a fresh reader (human or LLM)

**Status:** design-stage repository; no code yet. This document plus the three design
docs below contain all decisions; nothing else is assumed.
**License:** MIT for all code in this repository (see §3).

---

## 1. Documentation map (reading order)

| Document | Owns |
|---|---|
| **`REPO-CONTEXT.md`** (this) | Facts, toolchain, version pins, license, verified-facts ledger, references |
| **`nn-suite-architecture.md`** | Binding architecture: kernel-first layering, packaging, widget inventory, testing strategy, build order, cross-cutting non-choices |
| **`image-feature-extractor-widget.md`** | Image domain: ~35 interpretable features in six groups, evaluation protocol, experiments |
| **`tiny-lm-suite-design.md`** | Language-model domain: three-stage pre-training/LoRA teaching arc, four LM widgets, evaluation spine |

A fresh LLM should read this document first, then the architecture doc, then whichever
domain doc it is working on. Where documents overlap, the architecture doc wins on
cross-cutting decisions; domain docs win on domain detail.

---

## 2. Project purpose (one paragraph)

Educational tooling for teaching neural networks and LLMs, built as **`nn-kernel`** (a
pure-Python, Orange-free kernel: image features; tiny language models with Keras; LoRA
with MLX) plus **`orange3-nn-suite`** (one Orange add-on, category "Neural Networks", six
thin widget shells over the kernel). Classroom hardware: **16 GB Mac Mini (Apple
Silicon)**. Everything lives in Orange where possible; a future Python-only teaching
stream reuses the same kernel. No code has been written; the four action items at the
end of `nn-suite-architecture.md` (§8) are the pre-implementation gates.

---

## 3. License

**MIT** for all new code and documentation in this repository.

Two related cautions:

1. **`orange3-imagenets` is CC-BY-NC-SA (non-commercial).** It is the *pattern source*
   for the "QThread + Keras callback + live pyqtgraph loss curve" widget design cited
   throughout the docs. Being **inspired by the pattern is fine; copying its code is
   not** — it would infect this repository. Write widget code from scratch.
2. **`orange3-nnviewer` folds into this suite** (per the architecture doc). When it
   does, normalize its license to MIT as part of the fold (verify what it currently
   carries before publishing).

---

## 4. Toolchain conventions

Migrated from the proven `orange3-nnviewer` workflow; keep them identical so the
add-on builds the same way.

- **Build:** `uv build` → a single self-contained wheel.
- **Install target:** the Python environment that runs Orange — for the macOS
  Orange.app bundle, `/Applications/Orange.app/Contents/MacOS/python`. Install with
  `uv pip install --python <that interpreter> --no-deps dist/…whl`; re-resolving
  dependencies inside the bundled framework Python can trigger source builds and
  must be avoided.
- **Development:** editable install (`uv pip install --python … -e .`); edits are
  live on Orange restart.
- **Run canvas:** `python -m Orange.canvas`.
- **Widget-only preview during development:** Orange's `WidgetPreview`
  (`python -m orangecontrib.<pkg>.widgets.<module>`), as `orange3-nnviewer` does.
- **Tests:** `python -m pytest -q` with the Orange interpreter — Qt renders offscreen;
  no display needed. Kernel tests additionally must run in a *plain* Python (no
  Orange installed) to prove the kernel's Orange-freedom.
- **Publish:** `uv publish` (token via `UV_PUBLISH_TOKEN`); users then install from
  Orange's own UI (*Options → Add-ons… → Add more…*).

---

## 5. Version targets (pinned)

Minimum supported floors, matching the community add-on that is the closest packaging
precedent (`orange3-imagenets`, whose dependency set is proven inside Orange widgets).
**Release packaging pins exact versions** (lockfile/`requires-pin`), never ranges,
because both the Keras-in-Orange and MLX tuner surfaces move.

| Component | Minimum | Notes |
|---|---|---|
| Python | 3.11 | Ecosystem floor; Apple Silicon CPython |
| Orange3 | 3.39.0 | With `orange-canvas-core >= 0.1.28`, `orange-widget-base >= 4.20.0` |
| Keras (`[lm]` extra) | 3.10.0, with `tf_kera >= 2.19.0` | Proven in-widget combo; Keras 3 multi-backend is otherwise invisible at this scale |
| OpenCV (`[image]` extra) | 4.12 | `opencv-python-headless` |
| MLX (`[lora]` extra) | exact pin, recorded at first successful BritLLM dry run | `mlx-lm[train]`; **policy: exact pin only** — the tuner internals are less stable than Keras `fit` |
| Target OS | macOS, Apple Silicon | Classroom: 16 GB Mac Mini; MLX is Apple-only by design; non-Apple machines get a graceful "extra not available" path |

---

## 6. Qt discipline

**Use `AnyQt` consistently in all widget code. Never import raw `PyQt5`/`PySide`
directly.** (`orange3-imagenets` mixes `PyQt5` and `AnyQt` imports — a known latent
bug in that add-on and a bug we must not copy, along with its code, per §3.)

---

## 7. Reference implementations

| Reference | Role | Caution |
|---|---|---|
| [`orange3-nnviewer`](https://github.com/tcorbettclark/orange3-nnviewer) (ours) | In-house thin-shell pattern: widget boilerplate, `uv` workflow, offscreen tests, `--no-deps` install. **Its README content migrates into this document's §4 when the repo folds in.** | Normalize license to MIT |
| [`orange3-imagenets`](https://github.com/chrislee35/orange3-imagenets) (PyPI `orange3-imagenets` v0.0.3) | Pattern source only: `OWWidget` + Keras `Callback` streaming loss to pyqtgraph; QThread preprocessing; prebuilt-model dropdowns; save/load `.h5` | **CC-BY-NC-SA — no code copying.** Also known defects to avoid: no validation split, an unused dropout setting, mixed Qt imports, absolute thresholds |
| [`orange3-nlp`](https://github.com/chrislee35/orange3-nlp), [`orange3-ollama`](https://github.com/chrislee35/orange3-ollama) (same author) | Ecosystem context: what exists (embedders, RAG, Doc2Vec training, Ollama inference) | Experimental; nothing is reused |
| [Orange3-Text](https://orange3-text.readthedocs.io/) | The evaluation spine (Word Enrichment, Word Cloud, Concordance, Topic Modelling, Document Embedding/Map) — all existing widgets, used as-is | None |

---

## 8. Verified-facts ledger

Facts established by verification during design; re-verify anything older than a few
months before relying on it in code.

| Fact | Status | Source |
|---|---|---|
| `mlx-lm` is a Python library (CLI is a thin wrapper); trainer exposes a callback mechanism suitable for live loss streaming in a widget | Verified | [mlx-lm README, Python API section](https://github.com/ml-explore/mlx-lm) |
| MLX LoRA supports specific families: Mistral, Llama, Phi2, Mixtral, Qwen2, Gemma, OLMo, MiniCPM, InternLM2 — **GPT-2 is not among them** | Verified | [mlx_lm/LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| `mlx_lm.lora` supports **prompt masking** (loss on completion tokens only; `chat`/`completion` datasets) | Verified | [mlx_lm/LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| QLoRA is used automatically when the model is quantized; 4-bit quantization is the Stage-3 default | Verified | LORA.md, as above |
| BritLLM `britllm/britllm-3b-v0.1`: 3B params, F16, **raw base model** ("should be further finetuned"), Llama-family (community GGUF conversions exist) | Verified | [HF model card](https://huggingface.co/britllm/britllm-3b-v0.1), [llm.org.uk](https://llm.org.uk/) |
| **BritLLM has an existing MLX conversion** | **Assumed NO — action item** | None; dry run required (`nn-suite-architecture.md` §8) |
| Fallback: `britllm/TransWebLLM` (1B, same organisation) exists | Verified | [britllm HF org](https://huggingface.co/britllm) |
| TensorFlow's `tensorflow-metal` plugin is effectively unmaintained → Stage 1 trains CPU-only on Apple Silicon | Established at design time | Re-check at implementation |
| Keras levels 1–3 (`Sequential`, `Embedding`, `Dense`, `LSTM`/`GRU`) are stock; `MultiHeadAttention` exists but there is **no stock Transformer decoder block** in core Keras | Established | Keras docs |
| Orange's built-in Neural Network widget is a scikit-learn MLP on feature vectors; Orange cannot train CNNs/LLMs in the GUI — the gap this suite fills | Established | Orange docs |
| `orange3-imagenets` dependency set (Orange ≥ 3.39, Keras ≥ 3.10, `tf_kera` ≥ 2.19, OpenCV ≥ 4.12) works inside Orange widgets — the basis of §5's floors | Verified | PyPI metadata for `orange3-imagenets` v0.0.3 |
| Orange3-Text ships Word Enrichment and the analysis widgets the evaluation spine needs | Verified | [Orange3-Text docs](https://orange3-text.readthedocs.io/) |

---

## 9. Known constraints to carry into any implementation

- 16 GB unified memory on classroom machines: 4-bit QLoRA only (3B fp16 ≈ 6 GB before
  optimizer state); Stage-1 models sized for CPU minutes.
- The kernel never imports Orange or Qt (architecture doc §2, disciplines).
- The `[lora]` extra is optional with graceful degradation on non-Apple machines.
- Evaluation and visualization use **existing** Orange widgets only; this suite builds
  no new ones (NN Viewer excepted, already existing).