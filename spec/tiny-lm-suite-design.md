# Tiny LM Suite — teaching pre-training and fine-tuning inside Orange

**Status:** design doc (no code yet)
**Context:** educational use; students on 16 GB Mac Minis (Apple Silicon, no NVIDIA GPU)
**Companion documents:** `nn-suite-architecture.md` (kernel-first layering, packaging, and build order — binding for all suite components); `image-feature-extractor-widget.md` (same suite, image domain); general discussion of image/text classification in Orange (Image Embedding / Orange3-Text)

---

## 1. Goal

Teach how LLMs are pre-trained and fine-tuned, hands-on, with every artifact
inspectable rather than magic. Students:

1. watch a tiny model learn from scratch (loss fall, gibberish → domain pastiche),
2. confront what a real pretrained model does and doesn't know (raw base model),
3. efficiently adapt that model to a single domain (LoRA), experiencing both
   continued pre-training and true supervised fine-tuning (SFT),
4. quantify the single-domain limitation at every stage with Orange's text-mining stack.

Everything that *can* live in Orange does live in Orange. The terminal is not used.

---

## 2. The three-stage arc

```
Stage 1: Pre-training from scratch        Stage 2: What scale buys       Stage 3: Efficient adaptation
(Orange widgets + Keras)                  (Orange, inference only)       (Orange widget + MLX)
──────────────────────────────            ─────────────────────────      ──────────────────────────────
Tiny model trains on a domain             BritLLM (3B raw base model)    Same BritLLM model, LoRA-
corpus; live loss curve; samples          generated in Orange on the     tuned on domain data with
evolve gibberish → domain pastiche        same prompts; fluent text     ~0.1% of parameters moving;
                                          but shallow in the domain,    two beats: continued pre-
                                          poor instruction-following    training, then true SFT
```

Each stage fixes the previous stage's deficiency:

- Stage 1 ends at domain-flavored pastiche → Stage 2 supplies real English, but no domain
  depth and no instruction-following (it is a raw base model — that is the point).
- Stage 3 supplies domain depth (continued pre-training) and instruction-following
  (SFT) — while revealing new limitations: confidently domain-flavored hallucinations,
  general knowledge preserved but domain claims unverifiable.

Stages 2 and 3 deliberately use the **same model** so the before/after comparison is
perfectly clean: identical model, adapter on vs. off.

---

## 3. Architecture decision (Stage 1) — Keras levels 1–3

### Why Keras

- The widget plumbing pattern is proven in the community (`orange3-imagenets`:
  `model.fit` with a custom `Callback` streaming per-epoch loss into a live pyqtgraph
  plot, QThread for background work, `.h5`/JSON serialization).
- All chosen architectures are stock Keras layers; consistency with the existing
  Orange/Keras ecosystem means users install one heavy stack.
- PyTorch/MPS or MLX at Stage-1 scale would be faster but fragments the stack for no
  pedagogical gain. **Documented non-choice.**

### Why not a transformer in Stage 1

At our scale (≤ 1 MB corpus, ~100k–1M params, laptop CPU) attention offers no accuracy
advantage over an LSTM, and costs the Functional API, custom assembly, LR warmup, and
CPU training times that strain a classroom timeslot. More importantly, the transformer
lesson is delivered better by Stages 2–3, where students meet a real transformer and
open its hood via LoRA. The Trainer therefore stops at Keras levels 1–3. **No tiny-GPT
option anywhere in v1.**

### The architecture ladder (dropdown in the Trainer)

| Tier | Architecture | What it is | What it teaches |
|---|---|---|---|
| 0 | **Bigram NN** | Embedding → softmax over next token (context of 1) | The floor: pronounceable garbage; what *any* learning achieves |
| 1 | **MLP-over-window** (flagship) | Embed last k tokens (k≈8–16) → concatenate → hidden tanh → softmax | Context helps; memory is a hard, visible window |
| 2 | **LSTM** (1–2 layers, 128–256 units) | Recurrent; compressed hidden state carried forward | Memory cells; smooth training; sequential processing |

Running the tiers against each other on the same corpus and overlaying loss curves and
samples is the core Stage-1 experiment: the bigram plateau, the window limitation, the
LSTM's smoothness — the history of the field in three dropdown options.

Default: MLP-over-window (the "simple, but not too simple" centerpiece). LSTM runs are
"launch and return" exercises (10–30 min on CPU for good results).

### MLP-over-window specifics

- Tokenize → build all rolling (context, next-token) pairs; standard Keras utilities
  for windowing (`timeseries_dataset_from_array`) and tokenization (`TextVectorization`).
- Sequence-start handling: BOS/padding token so early tokens with short context are
  trainable; **windows do not cross sentence/document boundaries** (the model must not
  learn to continue across unrelated sentences).
- Train/inference asymmetry surfaced in the UI: training uses ground-truth next tokens;
  generation feeds its own predictions back.

---

## 4. Component design

Four widgets. The Trainer/Generator pair is used for Stage 1 (Keras models); the
Generator is reused for Stages 2–3 (MLX models). All training runs in a background
QThread with live loss curves — the established pattern.

**Layering:** all substance — tokenizer, windowing, model builders, sampling loop,
LoRA wrappers, and the model-object conventions (`generate` interface,
tokenizer-travels-with-model) — lives in the `nn-kernel` package (`nn_kernel.lm`,
`nn_kernel.lora`): pure Python, no Orange imports. The widgets are thin shells
(options, live plots, Orange-type adapters). See `nn-suite-architecture.md`.

### 4.1 LM Trainer (Keras)

- **Input:** Corpus. **Output:** model object (see tokenizer section) + checkpoint table.
- **Options:** architecture dropdown (bigram / MLP-window / LSTM); context window size;
  embedding dim; hidden/LSTM units; optimizer Adam; train/val split (~10%, mandatory —
  train/val divergence *is* the overfitting lesson on a tiny corpus).
- **Main area:** live training/validation loss **and perplexity (= e^loss, more
  interpretable: "the model is equivalently choosing among N plausible next characters")**;
  checkpoint sample table (epoch → sample text), readable in Corpus Viewer so students
  watch the text evolve from gibberish to domain pastiche.
- Tokenizer options (char-level vs. word-level) as an explicit experiment: vocabulary
  size, sequence length, loss curves — visible effects.

### 4.2 LM Generator

- **Input:** model (Keras or MLX — any object exposing `generate(prompt, temperature)`).
- **Options:** prompt; temperature slider (low = repetitive, high = word salad — the most
  instructive knob in LM demos); max length.
- **Output:** generations as a Corpus.
- The autoregressive sampling loop (~20 lines) is embedded in the model object — no
  library provides it; Keras models and MLX models both implement the same interface.

### 4.3 LoRA Trainer (MLX)

- **Input:** Corpus (raw text) or Table (prompt–response pairs); model selector
  (BritLLM). **Output:** adapter artifact (base-model reference + adapter files,
  serialized as one unit, same travels-with-model philosophy).
- Built on the **MLX Python API** (importable; the `mlx_lm` package is a library, not a
  CLI-only tool). Version pinned in the add-on; internals of the tuner are less stable
  than Keras's `fit`, so pinning is a maintenance commitment.
- **Quantization: 4-bit QLoRA only** — simplicity and the 16 GB constraint (3B fp16 =
  ~6 GB weights before optimizer state and activations; 4-bit fits comfortably). No
  full-precision option exposed.
- **Two data formats = two distinct lessons, one mechanism:**
  1. **Raw domain corpus → continued pre-training.** Still self-supervised (next-token
     loss; the data is its own supervisor). Teaches efficient adaptation of a pretrained
     backbone. LoRA dissected explicitly: W frozen, adapters W + BA, rank, the printed
     trainable-parameter count (~0.1% of the model — a number worth gasping at).
  2. **Prompt–response pairs → true SFT.** Genuinely supervised: x and y are separate
     things authored by a human, encoding intent. The pairing — not the loss, which is
     identical next-token cross-entropy — is what makes it supervised.
     **Prompt masking:** SFT computes loss on the response tokens only, not the prompt
     (MLX supports this natively for chat/completion datasets). Without it, the model
     wastes capacity learning to reproduce prompts.
  - **The "supervised or not?" quiz** is an explicit teaching beat: a curated list of
    500 domain sentences fed to LoRA — supervised or not? (No: chunking text into
    sentences changes document boundaries, not supervision. Supervision lives in the
    data *format*, not the training mechanism.)
- **Apple-only dependency handled deliberately:** optional install; graceful
  "MLX not available" message on non-Apple-Silicon machines. Uniform classroom hardware
  makes this acceptable.
- **LoRA support is model-specific:** verified families only. GPT-2 is not supported by
  the MLX toolchain — **documented non-choice** (also on tokenizer age grounds).

### 4.4 Vocab Inspector

- No standard Orange widget exists for this — we build it. Read-only; no pipeline role;
  no outputs that feed back into training/generation (prevents vocabulary-mismatch bugs
  by construction).
- **Input:** model (preferred) or corpus. **Shows:** token→id table; vocabulary size;
  token frequency distribution (Zipf's law — a side lesson); longest tokens; char-level
  vs. word-level side by side.
- Makes the Trainer's tokenizer dropdown a testable hypothesis ("char-level: smaller
  vocab, longer sequences") verified here and in the loss curves.

### 4.5 The tokenizer travels with the model

The vocabulary is a constituent part of the model, not a sibling: embedding row *i*
means token *i* only because the trainer's vocab said so. The Trainer builds the
tokenizer from the corpus and **embeds it in the emitted model object**; the Generator
uses `model.tokenizer` for encoding prompts and decoding samples; Save/Load serializes
vocab + weights as one unit. Rationale: a separate tokenizer widget would enable the
exact mismatch bug it seems to guard against (wiring a wrong vocab into the Generator),
and a tokenizer change requires retraining anyway. Hugging Face and GPT-2 both bundle
tokenizer with weights for the same reason.

---

## 5. Stage 2–3 model: BritLLM

**`britllm/britllm-3b-v0.1`** (UCL-led UK initiative, llm.org.uk): 3B parameters,
Llama-family architecture, **raw base model** ("should be further finetuned for most
use cases").

Chosen because:

- **Raw base model** is pedagogically essential: Stage 2 shows fluent text with poor
  instruction-following (it has never been SFT'd); Stage 3's SFT beat then has an
  observable payoff — instruction-following *appears*. The concept
  "pre-training ≠ instruction ability; SFT adds it" becomes behavior, not theory.
- Llama-family architecture ⇒ supported by the MLX LoRA toolchain.
- Single model across Stages 2–3 keeps the comparison clean and removes any other
  inference dependency (no KerasHub, no GPT-2).

**Action items (pre-implementation):**

- Verify/produce an MLX conversion of BritLLM (Llama-family ⇒ supported; dry run required).
- Dry-run LoRA training time on a 16 GB Mini; MLX versions and quantization choices shift
  timings (expect tens of minutes, not hours, on a small corpus).
- Pin `mlx-lm[train]` exact version inside the kernel's `[lora]` extra (policy: exact
  pins only — the tuner internals are less stable than Keras `fit`).
- **Fallback model** if the BritLLM conversion fails or 3B QLoRA proves too slow on
  the Mini: `britllm/TransWebLLM` (1B, same organisation) or another small
  Llama-family raw base model. Constraint: must be an MLX-LoRA-supported family;
  raw base preferred (the pre-training ≠ instruction-ability lesson depends on it).

---

## 6. Evaluation spine (Orange3-Text, across all stages)

Generated text from Stages 1–3 is imported back as corpora and run through **one shared
analysis pipeline** — Orange is the evaluation spine of the whole course even though it
only *trains* in Stages 1 and 3:

| Question | Components |
|---|---|
| Does each model's vocabulary match the domain? | Word Cloud; **Word Enrichment** (over/under-representation vs. the source corpus and bundled general corpora — the single-domain-bias detector) |
| Where does the domain sit in semantic space? | Document Embedding + Document Map (domain docs vs. general docs) |
| Does the model produce domain-plausible collocations that never occur in the source? | Concordance — the hallucination fingerprint, for every model |
| How limited is each model off-domain? | Perplexity tables (in-domain vs. out-of-domain, all stages) from the trainers/generators, visualized in Orange — the single-domain limitation as a chart spanning the course |
| How does the tiny model compare to a real LLM? | Side-by-side generations on identical prompts, all stages, in Corpus Viewer |

---

## 7. Corpus

Single-domain, ~0.5–1 MB, recognizable: Shakespeare (the classic; ~1 MB; instant
recognition), medical abstracts, legal text, or cooking recipes — all small and comically
biased in a way students will enjoy discovering. Orange's bundled text widgets (Guardian,
NY Times, Wikipedia, PubMed) provide ready contrast corpora for Word Enrichment.

Corpus sizing is part of the design (16 GB Mini, CPU-only training): ~500 KB–1 MB of
char-level text is the sweet spot.

---

## 8. Framework allocation

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Framework | Keras (levels 1–3) | (inference) | MLX |
| Why | Standard stack; proven widget pattern; CPU-trainable at tiny scale | — | Apple-Silicon-native (Metal GPU, unified memory); LoRA tooling |
| Constraint | 16 GB Mini = CPU-only TensorFlow training (no CUDA; Metal plugin for TF effectively abandoned) → minutes-scale models only | light | 4-bit QLoRA; optional dependency; Apple-Silicon-only |

The framework switch is stated to students explicitly as a design lesson: different
jobs, different tools — educational scale in the canvas, practitioner scale in the
widget wrapping a native Apple framework. The whole course remains in one environment.

---

## 9. Hardware fit (16 GB Mac Mini, Apple Silicon)

| Component | Time | Memory | Verdict |
|---|---|---|---|
| Bigram NN | seconds | negligible | perfect |
| MLP-over-window | minutes | < 500 MB | ideal flagship |
| LSTM | 10–30 min for good results | < 1 GB | "launch and return" exercises |
| BritLLM inference (4-bit) | fast | ~2 GB | fine |
| BritLLM QLoRA (4-bit) | tens of minutes | comfortable in 16 GB shared | classroom-viable; dry run required |

---

## 10. Roadmap

1. Corpus selection and preparation (domain corpus + contrast corpora).
2. `nn_kernel.lm` prototype: tokenizer first (Vocab Inspector logic early — it
   debugs the tokenizer), then windowing, bigram/MLP-window/LSTM builders,
   sampling loop; pytest throughout.
3. `LM Trainer` + `LM Generator` widget shells over the kernel.
4. MLX integration: verify BritLLM conversion; `nn_kernel.lora` with both data
   formats; dry-run timings on a 16 GB Mini; `LoRA Trainer` shell.
5. Student SFT dataset exercise design (authoring prompt–response pairs from the
   domain corpus).
6. Evaluation spine: workflows for the cross-stage comparison; perplexity tables.
7. Run the full three-stage arc end-to-end; record findings; tune timescales.

See `nn-suite-architecture.md` for the kernel-first layering and the cross-suite
build order.

---

## 11. Documented non-choices

| Not chosen | Why recorded |
|---|---|
| Tiny-GPT in the Trainer | No accuracy advantage at this scale; transformer lesson delivered better by Stages 2–3 with a real model |
| GPT-2 anywhere | Not LoRA-supported by the MLX toolchain; ancient tokenizer; poor stage-2 mascot |
| Full fp16 LoRA | 6 GB weights before optimizer state; 4-bit only, for simplicity and 16 GB headroom |
| PyTorch/MPS for Stage 1 | Faster, but fragments the stack for no pedagogical gain at Keras levels 1–3 |
| Separate tokenizer widget in the pipeline | Enables vocabulary-mismatch bugs; tokenizer change requires retraining anyway; inspector is read-only instead |
| Fine-tuning via a terminal session | MLX is a Python library; the widget integrates it directly — the whole course stays in Orange |
| Supervising signal from "curated sentences" | Whole sentences are still unlabeled text; supervision lives in data format (prompt–response pairs), not corpus curation |