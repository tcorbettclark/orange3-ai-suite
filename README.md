# orange3-nn-suite

Teaching neural networks and language models inside
[Orange](https://orangedatamining.com/). The add-on provides interpretable
image features, tiny Keras language models, and MLX LoRA fine-tuning in one
"Neural Networks" toolbox category.

Everything that can be inspected is inspectable: feature values are ordinary
Table columns, training curves are live, tokenizers travel with their models,
and generations come back as corpora that Orange's text widgets can analyse.

## What is in the box

- **`nn_kernel/`** - the substance. Pure Python, with no Orange or Qt imports:
  image features, tokenizers and windowing, Keras model builders, sampling,
  perplexity, and MLX LoRA wrappers. The same code runs in the widgets or in a
  plain Python session.
- **`nn_suite/`** - the Orange add-on. Eight widgets and the signal adapters that
  convert between Orange objects and kernel types. The widgets are thin shells:
  options, thumbnail grids, live plots and Orange signals.
- **`demo/`** - reproducible corpora, data fetchers and generated `.ows`
  workflows.

One wheel, one install, one category.

## Installation

Requirements: Python 3.11+, Orange 3.39+ for the widget layer, and macOS on
Apple Silicon for LoRA training. Optional dependencies are grouped per
capability:

| Extra | Brings | Used by |
|---|---|---|
| `image` | OpenCV | Image Feature Extractor |
| `lm` | Keras / TensorFlow | LM Trainer, LM Generator, Vocab Inspector |
| `lora` | `mlx-lm[train]` (Apple Silicon only) | LoRA Trainer |

Development (kernel tests run without Orange):

```bash
uv sync --extra lm
uv run pytest
```

Install into the Orange environment. On macOS with the Orange.app bundle:

```bash
uv pip install --python /Applications/Orange.app/Contents/MacOS/python \
    --no-deps -e . --no-build-isolation
/Applications/Orange.app/Contents/MacOS/python -m Orange.canvas
```

Widget tests use the Orange interpreter and render offscreen:

```bash
/Applications/Orange.app/Contents/MacOS/python -m pytest -q
```

Build and install a wheel:

```bash
uv build
uv pip install --python /Applications/Orange.app/Contents/MacOS/python \
    --no-deps dist/orange3_nn_suite-*.whl
```

## Widgets

| Widget | Input | Output | Extra |
|---|---|---|---|
| Image Feature Extractor | Images | Features, Selected image (Table) | `image` |
| Pixel Features | Images | Features (Table) | `image` |
| LM Text | none | Data (Table) | none |
| LM Trainer | Data (text) | Model, Checkpoint (Table) | `lm` |
| LM Generator | Model | Generations (Table) | `lm` / `lora` |
| LoRA Trainer | Data (text or pairs) | Model | `lora` |
| Vocab Inspector | Model or Data | none (read-only) | `lm` |
| NN Viewer | Model | none (read-only) | none |

Each widget class has a docstring describing its inputs, outputs and controls.

### Image Feature Extractor

Computes a small set of human-readable features from each image and emits one
row per readable image. Class variables and meta columns are inherited, and the
image path is kept as a meta column so any row can be traced back to its source.

Controls cover preprocessing (working size, centre crop, contrast
normalisation, optional median denoising) and feature groups A-G. Changes apply
automatically; type a working size and press Enter to reprocess. Toggling
groups is the ablation experiment; group G (colour) is the opt-in shortcut
demonstration. The main area shows every input image as a thumbnail, with the
total count and the current selection in the header. Thumbnails are
center-cropped to squares so wide and tall images fill the grid. The right-hand
panel shows
the selected image's feature values, grouped A-G, and a class-split histogram
for the selected feature. Disabled groups stay visible but greyed out. The
`Selected image` output carries the selected row as a one-row table, so it can
drive a data table's subset input. The peak-prominence and edge-density
thresholds are available as `FeatureOptions` fields in the kernel, not as
widget controls.

The output feeds any Orange learner. Test & Score, Confusion Matrix, Column
Statistics and Scatter Plot then make every stage of the pipeline auditable.

### Pixel Features

The deliberately plain baseline. Every image is converted to luminance,
resized to a square grid and flattened row-major into values in [0, 1]. The
only control is the grid dimension; the default 10 produces 100 features per
image. It performs no feature engineering and adds no spatial encoding, so it
is useful for comparing a raw-pixel MLP against the interpretable feature
groups or a convolutional model.

### LM Text

Reads UTF-8 `.txt` files from one file or a recursively scanned folder into a
plain Table. **Text documents** mode emits one row per whole file, paragraph,
line or sentence, with a `text` meta. **Q/A pairs** mode parses fixed blocks: a
new pair starts at `Q:`, the prompt includes the following `A:` line, and the
completion runs to the next `Q:`. The output uses `prompt` and `completion`
metas. The source file is shown in the preview but is not added to the Data
output.

Cleaning is opt-in: strip Project Gutenberg header/footer, reflow wrapped
lines, remove repeated short lines, and remove page-number lines. A small
preview table shows the first rows; selecting a row shows its full text below.

### LM Trainer

Pre-trains a small language model on a text corpus. The architecture dropdown
selects:

- **Bigram** - context of one token; the floor that any learning achieves.
- **MLP over a fixed window** - embeds the last *k* tokens and predicts the
  next; the default.
- **LSTM** - recurrent state; smoother but slower on CPU.

The tokenizer can be character-level or word-level, which visibly changes
vocabulary size and sequence length. Training always holds out a sentence-level
validation split, so overfitting on a tiny corpus is visible in the live loss
curve. The checkpoint output has one row per epoch with loss, perplexity and an
optional generated sample, so the text can be watched moving from gibberish to
domain pastiche.

The trained model embeds its tokenizer and can be saved as one directory:
`LanguageModel.save()` writes the Keras model, the vocabulary and the
hyperparameters together.

### LM Generator

Generates text from any model that exposes the kernel convention
`generate(prompt, temperature, max_new_tokens, seed)`. This covers Keras
`LanguageModel` objects and MLX `LoRAModel` objects, so the same widget serves
from-scratch models and fine-tuned base models.

The prompt, temperature, maximum new tokens, number of generations and seed are
all controls. Low temperature is repetitive; high temperature dissolves into
word salad. Generations are emitted as a one-column Table that Corpus Viewer
and Orange3-Text widgets can read. `Load model...` opens a saved Keras model or
LoRA adapter directory. For LoRA models, a **Use LoRA adapter** checkbox switches
between the adapted model and its raw base model.

### LoRA Trainer

Fine-tunes a pretrained model with 4-bit QLoRA on Apple Silicon. The training
data format is the lesson:

- **Raw corpus** - continued pre-training. The model is still self-supervised;
  the corpus supplies the next-token targets.
- **Prompt-response pairs** - supervised fine-tuning. A human authored the
  prompt and the completion; mlx-lm masks the prompt tokens so loss is
  computed on the completion only.

The live curve shows training and validation loss per iteration. The output is
a `LoRAModel`: the base model plus its adapter, serialised as one artifact. The
adapter can be switched off in the Generator to compare the same base model
before and after adaptation.

Only 4-bit quantised models are accepted. `demo/fetch_britllm.py` downloads and
converts the default model; a Hugging Face id is treated as unquantised because
its quantisation cannot be verified without downloading it.

### Vocab Inspector

A read-only view of a tokenizer. Connect a model to see the vocabulary embedded
in it, or connect text data to build character-level and word-level tokenizers
side by side. The summary reports vocabulary size, mean token length and the
fraction of tokens seen only once; the tabs list tokens, ids and corpus counts.
The inspector has no outputs, so it can never feed a mismatched vocabulary back
into training or generation.

### NN Viewer

Draws the Model output of Orange's own Neural Network widget (a scikit-learn
MLP). Nodes are arranged one column per layer; edge width encodes absolute
weight and colour its sign. Small networks print numeric weights on the edges;
hovering an edge highlights its endpoints and shows exact values. Biases appear
under the neurons they feed and in tooltips. The diagram scales to fit the
window and can be saved as a PNG.

## Tutorials

Hands-on walk-throughs live in [`tutorials/`](tutorials/README.md). They cover
the image workflow, tokenizers, training from scratch, generation, LoRA
continued pre-training and SFT, and evaluation with Orange. Each tutorial is
self-contained and can be used as a classroom handout or as a manual test
script.

## The teaching arc

The language-model widgets support three beats, each fixing a limitation the
previous one exposes.

### 1. Train a model from scratch

Use **LM Trainer** on a single-domain corpus loaded with **LM Text**, for
example a public-domain short story. Start with the bigram, then the
MLP-over-window, then the LSTM. Compare
loss curves and checkpoint samples: the bigram plateaus quickly, the MLP has a
fixed and visible memory, and the LSTM trains more smoothly at a higher cost.
Switch between character and word tokenization and watch the vocabulary and
sequence lengths change. Use **Vocab Inspector** to see the tokenizer behind
the curve.

The result is a model that produces domain-flavoured pastiche, not knowledge.

### 2. Meet a real base model

Open the same prompts in **LM Generator** with a pretrained base model and no
adapter. The text is fluent, but the model is a raw base model: it has not been
instruction-tuned, so it continues text rather than answering questions. This
is the point where scale buys fluency, not obedience.

### 3. Adapt the base model

Use **LoRA Trainer** on the same base model in two passes:

1. **Continued pre-training** on the raw domain corpus, which adds domain depth.
2. **Supervised fine-tuning** on authored prompt-response pairs, which makes
   instruction-following appear.

The adapter's trainable-parameter share is a small fraction of the base model's
parameters; the status text and the widget controls make that concrete. Switch
the adapter on and off in **LM Generator** and compare identical prompts.

Across all three beats, Orange is the evaluation spine: Word Cloud, Word
Enrichment, Concordance, Document Embedding and Document Map turn generations
into corpora and expose domain bias, collocations and off-domain behaviour.
Perplexity from the train and generation APIs can be tabulated in Orange as
well.

## The image arc

A typical canvas workflow:

```
Import Images -> Image Feature Extractor -> Test & Score -> Confusion Matrix
                                            -> Data Table
                                            -> Column Statistics
```

Experiments that the widgets make easy:

- **Group ablation** - switch feature groups off and watch accuracy move.
- **The colour shortcut** - enable group G, the deliberately naive colour
  features; "fraction of bluish pixels" can solve cloud detection without
  learning any spatial structure.
- **Model transparency** - run the same features through a neural network, a
  decision tree and logistic regression. If all three land within a couple of
  points, the features carried the intelligence, not the model.
- **Stability** - small shifts, jitter and denoising should not flip category
  features. The kernel has dedicated stability tests.

## Image feature groups

All features are defined in the group modules and collected by
`nn_kernel.features.registry`; each carries a human-auditable definition
sentence.

| Group | Count | What it describes |
|---|---|---|
| A | 6 | Global photometry: brightness, contrast, skewness, kurtosis, bright fraction |
| B | 12 | Row/column/diagonal profile summaries: trend, roughness, peaks, centre of mass, crossings |
| C | 6 | Spatial mass: centre of mass, spread, elongation, radial gradient |
| D | 6 | Texture: gradient, edge density, fine/coarse texture, autocorrelation width |
| E | 3 | Topology: hole count, foreground area, stroke crossings |
| F | 3 | Symmetry: left-right, top-bottom, quadrant similarity |
| G | 4 | Colour (opt-in): bluish fraction, blueness, saturation, greenery |

Groups A-F run on a preprocessed luminance image. Group G reads colour
directly and is off by default.

```python
from nn_kernel.features import all_features, extract_features, get_feature

vec = extract_features(rgb_image, groups="ABCDEF")
vec["hole_count"]                 # a feature value
vec.to_array()                    # one float per selected feature
get_feature("hole_count").definition   # the human-readable sentence
all_features()                    # the full catalogue, in stable column order
```

## Kernel API

### Language models

```python
from nn_kernel.lm import Tokenizer, make_windows, train_language_model

tokenizer = Tokenizer.from_corpus(texts, "char")
windows = make_windows(tokenizer, texts, context_size=16)
train, valid = windows.split(val_fraction=0.1, seed=0)

model = train_language_model(
    texts, tokenizer, "mlp_window",
    context_size=16, embedding_dim=32, hidden_units=128,
    epochs=15, validation_fraction=0.1, seed=0,
)
model.generate("the ", temperature=0.8, max_new_tokens=64)
model.perplexity(held_out_texts)          # {'loss': ..., 'perplexity': ...}
model.save("my-model")                    # weights + tokenizer + metadata
```

`LanguageModel.load(path)` reads the directory back as one object. The
training hook (`on_epoch`) and the stop callback (`stop_requested`) are plain
Python callables; the kernel never imports Qt.

### LoRA

```python
from nn_kernel.lora import LoraData, LoraOptions, LoRAModel, train_lora

data = LoraData.from_texts(corpus)        # or LoraData.from_pairs(pairs)
adapter = train_lora(
    "data/models/britllm-4bit", data, "my-adapter",
    options=LoraOptions(iters=300, num_layers=16),
)
model = LoRAModel(adapter.base_model, adapter.adapter_path)
model.generate("In the second century", temperature=0)
model.adapter_enabled = False             # the same base model, before adaptation
```

`train_lora` runs the documented `python -m mlx_lm lora` command in a
subprocess, streams parsed metrics, and supports clean cancellation.

## Demo data and workflows

- `demo/mnist/` - committed subset of MNIST (500 images/class for training,
  100/class for validation), stored as PNGs in the folder-per-label layout that
  Import Images expects. Regenerate with `demo/fetch_mnist.py`.
- `data/text/` - optional downloaded plain-text corpora. No text corpus is
  committed; use **LM Text** to load any UTF-8 `.txt` file.
- `demo/fetch_clouds.py` - fetches a five-genus cloud corpus from the WMO
  International Cloud Atlas into `data/clouds/`. The images are untracked and
  remain the property of their photographers; they are not covered by this
  repository's code licence.
- `demo/fetch_britllm.py` - downloads `britllm/britllm-3b-v0.1` and converts it
  to a 4-bit MLX model in `data/models/britllm-4bit/`. The weights keep their
  own licence.
- `demo/mnist_features.ows`, `demo/cloud_features.ows` - ready-made canvas
  workflows. Regenerate them with `demo/make_workflows.py`; the Import Images
  paths inside are machine-local, so re-point the File widget if needed.

```bash
uv run python demo/fetch_mnist.py
uv run python demo/fetch_clouds.py
uv run python demo/fetch_britllm.py
uv run python demo/make_workflows.py
```

## Verification

- Kernel tests run in plain Python, with no Orange and no Qt. They are the
  proof that `nn_kernel` stays independent.
- Widget tests run in the Orange interpreter with Qt offscreen and cover the
  signal adapters, widget state, model persistence and the viewer's rendering.
- End-to-end model loads are marked `slow` and excluded by default.
- Lint and formatting are checked with ruff (configuration in
  `pyproject.toml`).

Run the full verification from the repository root:

```bash
uvx ruff check nn_kernel nn_suite demo tests conftest.py
uvx ruff format --check nn_kernel nn_suite demo tests conftest.py

uv run pytest -q
uv run pytest -q -m slow            # kernel, including slow model round trips

QT_QPA_PLATFORM=offscreen \
    /Applications/Orange.app/Contents/MacOS/python -m pytest -q   # + widgets
```

The kernel suite must pass in a plain Python environment; the widget suite must
pass in the Orange interpreter. One slow test is skipped in the default widget
run unless the converted MLX model is present.

## Layout

```
nn_kernel/
  features/   preprocessing, registry, groups A-G, extraction front door
  lm/         tokenizer, windowing, vocab statistics, Keras builders,
              training, LanguageModel with embedded tokenizer
  lora/       LoRA data formats, MLX training subprocess, artifact/model objects
nn_suite/
  adapters.py Orange <-> kernel conversion
  widgets/    the eight Orange widgets and their icons
demo/         corpora, fetchers, generated .ows workflows
tests/        kernel tests + offscreen widget tests
```

## License

MIT for the code in this repository.

Third-party material keeps its own terms: MNIST is publicly redistributed,
downloaded text keeps its own licence, WMO Cloud Atlas images remain the
property of their photographers and are used for personal, non-commercial
teaching, and BritLLM weights keep their model licence.
