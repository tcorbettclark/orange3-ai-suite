# orange3-ai-suite

Teaching neural networks and language models inside
[Orange](https://orangedatamining.com/). The add-on provides interpretable
image features, tiny Keras language models, and MLX LoRA fine-tuning across
three toolbox categories: **Language Models**, **Image Analytics** and
**Visualize**.

Everything that can be inspected is inspectable: feature values are ordinary
Table columns, training curves are live, tokenizers travel with their models,
and generations come back as corpora that Orange's text widgets can analyse.

## What is in the box

- **`orangecontrib/ai_suite/kernel/`** - the substance. Pure Python, with no
  Orange or Qt imports: image features, tokenizers and pairs, the architectures,
  the model that trains and generates, and MLX LoRA wrappers. The same code runs in
  the widgets or in a plain Python session.
- **`orangecontrib/ai_suite/`** - the Orange add-on. Nine widgets and the signal
  adapters that convert between Orange objects and kernel types. The widgets
  are thin shells: options, thumbnail grids, live plots and Orange signals.
- **`demo/`** - reproducible corpora, data fetchers and generated `.ows`
  workflows.
- **`GLOSSARY.md`** - the words the text and model code uses, one definition
  each: tokenizer, token, context, pair, model, sampler.

One wheel, one install, three toolbox categories.

## Installation

Requirements: Python 3.11+, Orange 3.39+ for the widget layer, and macOS on
Apple Silicon for LoRA training.

Everything except LoRA is a plain dependency: OpenCV for the image widgets,
Keras on a TensorFlow backend for the from-scratch language models. Only LoRA is
an extra, because MLX runs on Apple Silicon alone:

```bash
pip install orange3-ai-suite            # everything but LoRA
pip install "orange3-ai-suite[lora]"    # plus LoRA (Apple Silicon)
```

Development (kernel tests run without Orange):

```bash
uv sync
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
    --no-deps dist/orange3_ai_suite-*.whl
```

## Widgets

| Widget | Category | Input | Output | Needs `lora` |
|---|---|---|---|
| Image Features | Image Analytics | Images | Features, Selected image (Table) | no |
| Image Pixels | Image Analytics | Images | Features (Table) | no |
| LM Text Loader | Language Models | none | Data (Table) | no |
| LM Pre-trainer | Language Models | Data (text) | Model, Checkpoint (Table) | no |
| LM Generator | Language Models | Model | Generations (Table) | no |
| LoRA Pre-trainer | Language Models | Data (text) | Model | yes |
| LoRA SFT | Language Models | Data (prompt / completion) | Model | yes |
| LM Vocab Viewer | Language Models | Model or Data | none (read-only) | no |
| Neural Network Viewer | Visualize | Model | none (read-only) | no |

Each widget class has a docstring describing its inputs, outputs and controls.

### Image Features

Computes a small set of human-readable features from each image and emits one
row per readable image. Class variables and meta columns are inherited, and the
image path is kept as a meta column so any row can be traced back to its source.

Controls cover preprocessing (working size, centre crop, contrast
normalisation, optional median denoising) and feature groups A-G. Changes apply
automatically; type a working size and press Enter to reprocess. Toggling
groups is the ablation experiment; group G (colour) is the opt-in shortcut
demonstration. The main area shows every input image as a thumbnail, with the
total count and the current selection in the header. Thumbnails are
centre-cropped to squares so wide and tall images fill the grid. The right-hand
panel shows
the selected image's feature values, grouped A-G, and a class-split histogram
for the selected feature. Disabled groups stay visible but greyed out. The
`Selected image` output carries the selected row as a one-row table, so it can
drive a data table's subset input. The peak-prominence and edge-density
thresholds are available as `FeatureOptions` fields in the kernel, not as
widget controls.

The output feeds any Orange learner. Test & Score, Confusion Matrix, Column
Statistics and Scatter Plot then make every stage of the pipeline auditable.

### Image Pixels

The deliberately plain baseline. Every image is converted to luminance,
resized to a square grid and flattened row-major into values in [0, 1]. The
only control is the grid dimension; the default 10 produces 100 features per
image. It performs no feature engineering and adds no spatial encoding, so it
is useful for comparing a raw-pixel MLP against the interpretable feature
groups or a convolutional model.

### LM Text Loader

Reads UTF-8 `.txt` files from one file or a recursively scanned folder into a
plain Table. **Text documents** mode emits one row per whole file, per
paragraph or per sentence, with a `text` meta; the rows are storage, not the
model's units, which are always sentences. **Q/A pairs** mode parses fixed blocks: a
new pair starts at `Q:`, the prompt includes the following `A:` line, and the
completion runs to the next `Q:`. The output uses `prompt` and `completion`
metas. The source file is shown in the preview but is not added to the Data
output.

Cleaning is opt-in: strip Project Gutenberg header/footer, reflow wrapped
lines, remove repeated short lines, and remove page-number lines. A small
preview table shows the first rows; selecting a row shows its full text below.

### LM Pre-trainer

Pre-trains a small language model from scratch on a text corpus. The architecture dropdown
selects:

- **Bigram** - context of one token; the floor that any learning achieves.
- **MLP** - embeds the last *w* ids and concatenates them into one hidden
  layer, so its memory is fixed and visible; the default.
- **LSTM** - recurrent state; smoother but slower on CPU.

The tokenizer can be the character or the word one, which visibly changes
vocabulary size and sequence length. Training always holds out a sentence-level
validation split, so overfitting on a tiny corpus is visible in the live loss
curve. The checkpoint output has one row per epoch with loss, perplexity and an
optional generated sample, so the text can be watched moving from gibberish to
domain pastiche.

The trained model embeds its tokenizer and can be saved as one directory:
`LanguageModel.save()` writes the Keras model, the vocabulary and the
hyperparameters together. The widget itself has no Save button; persistence is
moving to dedicated Load LM/Save LM widgets.

### LM Generator

Generates text from any model that exposes the kernel convention
`generate(prompt, temperature, max_new_tokens, seed)`. This covers Keras
`LanguageModel` objects and MLX `LoRAModel` objects, so the same widget serves
from-scratch models and fine-tuned base models.

The prompt, temperature, maximum new tokens, number of generations and seed are
all controls. Low temperature is repetitive; high temperature dissolves into
word salad. Generations are emitted as a one-column Table that Corpus Viewer
and Orange3-Text widgets can read. For LoRA models, a **Use LoRA adapter**
checkbox switches between the adapted model and its raw base model. Model
persistence is intentionally not part of the training or generation widgets; a
dedicated Load LM/Save LM pair will own that role.

### BritLLM Models

Downloads and caches the base model the LoRA tutorials start from: a pinned
4-bit MLX conversion of `britllm/britllm-3b-v0.1` (2.67B parameters, 1.4 GB).
Press **Download**; the progress bar reads bytes written into the cache. The
model goes into the standard Hugging Face hub cache (`HF_HUB_CACHE`, else
`$HF_HOME/hub`); set `HF_HOME` to share a lab cache. The **Model path** output
is the local snapshot directory - connect it to **Base model** on LoRA
Pre-trainer or LoRA SFT. The LoRA widgets do not look the path up themselves.

Start with this 4-bit model. The Base model field also accepts other local MLX
model directories or Hugging Face ids; 8-bit or unquantised bases work if the
machine has the memory.

Downloads use the classic Hugging Face LFS transfer by default: it gives a
resumable file and a byte-accurate progress bar. Set `HF_HUB_DISABLE_XET=0`
before launching Orange to opt into Xet if you want chunk-level dedup.

### LoRA Pre-trainer

Continues pre-training a pretrained model with LoRA (QLoRA with the 4-bit
BritLLM) on Apple Silicon. Input is a raw text corpus, one document per row.
The model is still self-supervised: the corpus supplies the next-token targets,
and the adapter only restricts which weights change. Use this for domain
adaptation, not for teaching a response pattern.

The live curve shows training and validation loss per iteration. The output is
a `LoRAModel`: the base model plus its adapter. The adapter can be switched off
in the Generator to compare the same base model before and after adaptation.

### LoRA SFT

Supervised fine-tuning with LoRA (QLoRA with the 4-bit BritLLM) on Apple
Silicon. Input is a Table with `prompt` and `completion` text columns. A human
authored the prompt and the desired completion; mlx-lm masks the prompt tokens
so loss is computed on the completion only. The prompt/response contract covers
Q&A, instruction following and tool use; it is not tied to the `Q:`/`A:`
grammar of the text loader.

Both LoRA widgets share the same training controls, live loss curve and Model
output. The Base model input takes the path from BritLLM Models; the line edit
also accepts any local MLX model directory or Hugging Face id. The course uses
the 4-bit model; larger or unquantised bases need more memory.

### LM Vocab Viewer

A read-only view of a tokenizer. Connect a model to see the vocabulary embedded
in it, or connect text data to build the character and word tokenizers
side by side. The summary reports vocabulary size, mean token length and the
fraction of tokens seen only once; the tabs list tokens, ids and corpus counts.
The viewer has no outputs, so it can never feed a mismatched vocabulary back
into training or generation.

### Neural Network Viewer

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

Use **LM Pre-trainer** on a single-domain corpus loaded with **LM Text Loader**, for
example a public-domain short story. Start with the bigram, then the
MLP, then the LSTM. Compare
loss curves and checkpoint samples: the bigram plateaus quickly, the MLP has a
fixed and visible memory, and the LSTM trains more smoothly at a higher cost.
Switch between character and word tokenization and watch the vocabulary and
sequence lengths change. Use **LM Vocab Viewer** to see the tokenizer behind
the curve.

The result is a model that produces domain-flavoured pastiche, not knowledge.

### 2. Meet a real base model

Open the same prompts in **LM Generator** with a pretrained base model and no
adapter. The text is fluent, but the model is a raw base model: it has not been
instruction-tuned, so it continues text rather than answering questions. This
is the point where scale buys fluency, not obedience.

### 3. Adapt the base model

Use the two LoRA widgets on the same base model:

1. **LoRA Pre-trainer** on the raw domain corpus, which adds domain depth.
2. **LoRA SFT** on authored prompt-response pairs, which makes
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
Import Images -> Image Features -> Test & Score -> Confusion Matrix
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

Each feature is a class in `orangecontrib.ai_suite.kernel.images.features.catalogue`,
carrying its own name, description sentence and computation. Groups are a
set of membership lists over that flat catalogue
(`orangecontrib.ai_suite.kernel.images.features.groups`): membership lists used for
ablation and display, so a feature can belong to more than one group and the
grouping never touches the feature itself.

| Group | Count | What it describes |
|---|---|---|
| A | 6 | Global photometry: brightness, contrast, skewness, kurtosis, bright fraction |
| B | 10 | Row/column/diagonal profile summaries: trend, roughness, peaks, crossings |
| C | 6 | Spatial mass: centre of mass, spread, elongation, radial gradient |
| D | 6 | Texture: gradient, edge density, fine/coarse texture, autocorrelation width |
| E | 3 | Topology: hole count, foreground area, stroke crossings |
| F | 3 | Symmetry: left-right, top-bottom, quadrant similarity |
| G | 4 | Colour (opt-in): bluish fraction, blueness, saturation, greenery |

Groups A-F run on a preprocessed luminance image. Group G reads colour
directly and is off by default.

```python
from orangecontrib.ai_suite.kernel.images.features import (
    all_features,
    extract_features,
    get_feature,
)

vec = extract_features(rgb_image, groups="ABCDEF")
vec["hole_count"]  # a feature value
vec.to_array()  # one float per selected feature
get_feature("hole_count").description  # title, sentence, intuition, mechanics
all_features()  # the full catalogue, in stable column order
```

### The feature guide

Every feature carries four descriptions: a title, a one-sentence summary, a
plain-language account of which pictures move it and which leave it cold, and a
technical account of how it is computed with links for further reading. The
whole guide is generated from the catalogue when it is asked for, so it cannot
drift from the code:

```python
from orangecontrib.ai_suite.kernel.images.features import feature_guide_markdown

print(feature_guide_markdown())
```

In the widget, **Feature guide…** opens the same page, the selected feature's
title, sentence and intuition are shown beside its values, and **Technical
details…** opens the mechanics for that one feature.

What that plain baseline produces, cell by cell, is shown by
`examples/image_pixels.py` (`uv run python examples/image_pixels.py`); it prints
one photograph as a grid of numbers and as characters, with the column name of
every value.

A runnable script lives in `examples/image_features.py`
(`uv run python examples/image_features.py`). It reads three real cloud
photographs from `examples/` (credits in `examples/CREDITS.md`) and shows the
catalogue and its descriptions, the extraction and ablation selections, fine
texture on wispy cirrus against coarse texture on a roll cloud, and the colour
shortcut separating a blue sky from ground vegetation.

## Kernel API

### Language models

A runnable end-to-end script lives in `examples/lm_from_scratch.py`
(`uv run python examples/lm_from_scratch.py`); the essentials are:

```python
from orangecontrib.ai_suite.kernel.lm import (
    LanguageModel,
    MLP,
    TrainingOptions,
    make_pairs,
    tokenizer_from_text,
)

text = "\n".join(texts)  # join documents outside the kernel
tokenizer = tokenizer_from_text(text, "char")
tokenizer = tokenizer.keep_top(2000)  # or .drop_rare(min_count=3); pure, chainable
stream = tokenizer.encode(text)  # every sentence ends with <eos>
pairs = make_pairs(stream, context_width=16)
train, valid = pairs.split(validation_fraction=0.1, seed=0)

# An architecture is the recipe; a model is that recipe fitted.
architecture = MLP(context_width=16, embedding_dim=32, hidden_units=128)
model = LanguageModel(architecture, tokenizer).fit(
    text,
    TrainingOptions(epochs=15, validation_fraction=0.1, seed=0),
)

model.generate("the ", temperature=0.8, max_new_tokens=64)
model.evaluate(held_out_text)  # {'loss': ..., 'perplexity': ...}; perplexity = e^loss
model.save("my-model")  # weights + tokenizer + metadata
```

`fit` returns the model, so the call can be chained; it builds pairs at
`architecture.context_width`, so the data and the weights cannot disagree.
`LanguageModel.load(path)` reads the directory back as one object. The training
hook (`on_epoch`) and the stop callback (`stop_requested`) are plain Python
callables; the kernel never imports Qt.

### LoRA

```python
from orangecontrib.ai_suite.kernel.assets import get_britllm
from orangecontrib.ai_suite.kernel.lora import LoRAData, LoRAOptions, LoRAModel, train_lora

base = get_britllm()  # pinned 4-bit MLX weights, cached on first use
data = LoRAData.from_texts(corpus)  # or LoRAData.from_pairs(pairs)
adapter = train_lora(
    base,
    data,
    "my-adapter",
    options=LoRAOptions(iters=300, num_layers=16),
)
model = LoRAModel(adapter.base_model, adapter.adapter_path)
model.generate("In the second century", temperature=0)
model.evaluate(held_out_text)  # {'loss': ..., 'perplexity': ...}
model.adapter_enabled = False  # the same base model, before adaptation
```

`train_lora` runs the documented `python -m mlx_lm lora` command in a
subprocess and streams parsed metrics. Stopping terminates the whole child
process tree. mlx-lm writes a checkpoint every `save_every` iterations; a stop
adopts the newest checkpoint and records `iters_completed`, while a stop before
the first one raises `TrainingStopped` and produces no model.

## Datasets and workflows

```python
from orangecontrib.ai_suite.kernel.assets import get_clouds, get_mnist

mnist = get_mnist()  # <root>/train, <root>/val: 28x28, 500/100 per digit
clouds = get_clouds()  # <root>/ac, sc, cu, cb, ci + CREDITS.md
```

Both functions download, prepare and verify a pinned Hub dataset. The source
and the prepared tree are cached; repeated calls are offline.

- `MNIST_DIGITS` - [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)
  (MIT), 60k/10k parquet; `get_mnist()` keeps 500/100 per digit at the source
  28x28 resolution in the folder-per-label layout Import Images expects.
- `CCSN_CLOUDS` - [aduuuuuu/CCSN](https://huggingface.co/datasets/aduuuuuu/CCSN)
  (MIT mirror; the CCSN database itself is CC0), the five genera Ac, Sc, Cu,
  Cb and Ci at 256x256, with `CREDITS.md`. Cite Zhang et al. (2018), *CloudNet*.
- `data/text/` - optional downloaded plain-text corpora. No text corpus is
  committed; use **LM Text Loader** to load any UTF-8 `.txt` file.
- `examples/fetch_britllm.py` - fetches the pinned 4-bit MLX BritLLM into the
  Hugging Face cache (the headless form of the BritLLM Models widget).
- `examples/build_britllm_4bit.py` - maintainer script that downloads the fp16
  source, converts it, adds the chat template and (optionally) uploads the
  artifact. It builds in a temporary directory unless `--target` is given, so
  the repository is not written to.
- `demo/mnist_features.ows`, `demo/cloud_features.ows` - ready-made canvas
  workflows generated by `demo/make_workflows.py` from the prepared asset
  paths. The Import Images paths inside are machine-local; re-point the File
  widget if the cache moved.

```bash
orange-ai-assets fetch mnist-digits
orange-ai-assets fetch ccsn-clouds
orange-ai-assets fetch britllm-3b-v0.1-4bit-mlx
uv run --extra lora python examples/fetch_britllm.py
uv run python demo/make_workflows.py
```

## Verification

- Kernel tests run in plain Python, with no Orange and no Qt. They are the
  proof that `orangecontrib.ai_suite.kernel` stays independent.
- Widget tests run in the Orange interpreter with Qt offscreen and cover the
  signal adapters, widget state, model persistence and the viewer's rendering.
- End-to-end model loads are marked `slow` and excluded by default.
- Lint and formatting are checked with ruff (configuration in
  `pyproject.toml`).

Run the full verification from the repository root:

```bash
uvx ruff check orangecontrib demo tests conftest.py
uvx ruff format --check orangecontrib demo tests conftest.py

uv run pytest -q
uv run pytest -q -m slow            # kernel, including slow model round trips

QT_QPA_PLATFORM=offscreen \
    /Applications/Orange.app/Contents/MacOS/python -m pytest -q   # + widgets
```

The kernel suite must pass in a plain Python environment; the widget suite must
pass in the Orange interpreter. One slow test is skipped in the default widget
run unless the pinned 4-bit model is already in the Hugging Face cache.

## Reviewing a change

Structural changes - new modules, moved code, renames, splits - have a checklist
in [REVIEW.md](REVIEW.md): words, steps and boundaries, layout, evidence, and how
to finish. It is worth reading before choosing a shape as well as before merging
one.

## Layout

```
orangecontrib/
  ai_suite/
    __init__.py
    adapters.py Orange <-> kernel conversion
    kernel/
      images/     image data: decoded, resampled, then measured
        decode.py    RGB/luma, alpha and [0,1] conventions (shared)
        resize.py    one resampling policy: area shrinking, bilinear growing
        flatten.py   a grid as one row, with a coordinate name per cell
        features/    the feature package
          catalogue/   one module and one class per feature, plus the index
          description.py  the four descriptions, checked when a feature is created
          feature.py   the feature contract, checked at import
          guide.py     the whole catalogue as one generated Markdown page
          groups.py    groups as membership lists over the catalogue
          extraction.py  extract_features and FeatureVector
          options.py   FeatureOptions
          _preprocessing.py  the pipeline (support code)
          _conventions.py  numeric rules (zero, flat, net change)
          _measurements.py  measurements shared by more than one feature
      lm/         tokenizer (with its stats), pairs, architectures,
                  training options/results, LanguageModel with embedded tokenizer
      lora/       LoRA data formats, MLX training subprocess, artifact/model objects
      text/       plain text into training rows, behind one facade
        _clean.py    raw text -> clean text (Gutenberg, page numbers, reflow)
        _split.py    clean text -> table rows (whole file, paragraph, sentence)
        _qa.py       clean text -> Q/A prompt/completion pairs
    widgets/      the nine Orange widgets and their icons
demo/             generated .ows workflows and their generator
examples/         runnable kernel examples
tests/            kernel tests + offscreen widget tests
```

## License

MIT for the code in this repository.

Third-party material keeps its own terms: MNIST is MIT, the CCSN cloud
database is CC0, downloaded text keeps its own licence, and the 4-bit BritLLM
artifact keeps the source model's ODC-BY licence.
