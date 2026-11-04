# 00 - Setup

Get the add-on, the data and a text Table ready. Everything else in the
tutorials builds on this page.

## 1. Install and launch

Follow the project [README](../README.md#installation) to install the add-on
into the Orange interpreter. For the LoRA tutorials you also need the `[lora]`
extra in that interpreter and a converted BritLLM model:

```bash
uv pip install --python /Applications/Orange.app/Contents/MacOS/python \
    "mlx-lm[train]==0.32.2"
uv run --extra lora python demo/fetch_britllm.py
```

The development environment needs the `[lm]` extra for the MNIST builder and
the kernel model tests:

```bash
uv sync --extra lm
```

Launch the canvas from the repository root:

```bash
/Applications/Orange.app/Contents/MacOS/python -m Orange.canvas
```

The widgets are in the **Neural Networks** category.

## 2. Data

The committed demo data is enough for most tutorials:

- `demo/mnist/train/` - PNG digits in one subfolder per label;
- `demo/gibbon_corpus.txt` - public-domain Gibbon, Volume One.

The cloud corpus and the BritLLM model are optional. Fetch them when a tutorial
asks for them:

```bash
uv run python demo/fetch_clouds.py
uv run --extra lora python demo/fetch_britllm.py
```

## 3. Make a text Table with Python Script

The LM widgets accept an Orange Table with a string (text) column. Without
Orange3-Text, the simplest way to make one is the standard **Python Script**
widget:

1. Add **Python Script** from the widget list.
2. Paste one of the snippets below.
3. Press **Run**.
4. Connect its **Data** output to **LM Trainer**, **Vocab Inspector** or
   **LoRA Trainer**.

### Full Gibbon corpus

```python
import numpy as np
from pathlib import Path
from Orange.data import Domain, StringVariable, Table

text = Path("demo/gibbon_corpus.txt").read_text(encoding="utf-8")
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

domain = Domain([], metas=[StringVariable("text")])
out_data = Table.from_numpy(
    domain,
    np.empty((len(paragraphs), 0)),
    None,
    np.asarray(paragraphs, dtype=object).reshape(-1, 1),
)
```

### A few paragraphs

Use this when you want a fast experiment. Replace the list with any text you
like.

```python
import numpy as np
from Orange.data import Domain, StringVariable, Table

texts = [
    "The king gathered his counsellors in the great hall. The north wind "
    "carried tidings from the coast.",
    "The physician arrived at dusk, his satchel heavy with herbs. He bled "
    "the fevered patient and prescribed rest.",
    "The merchant counted his coins by candlelight, muttering of tariffs "
    "and the dishonesty of strangers.",
]

domain = Domain([], metas=[StringVariable("text")])
out_data = Table.from_numpy(
    domain,
    np.empty((len(texts), 0)),
    None,
    np.asarray(texts, dtype=object).reshape(-1, 1),
)
```

## 4. Open a prepared workflow

The repository ships two canvases:

- `demo/mnist_features.ows`
- `demo/cloud_features.ows`

Use **File > Open** and choose one. The Import Images paths inside are
machine-local. If a path is stale, open **Import Images**, click the folder
button and point it at `demo/mnist/train` or `data/clouds`.

If you prefer to build from scratch, every tutorial lists the widgets and
links.

## 5. Controls that apply automatically, and controls that do not

- **Image Feature Extractor**: preprocessing and group changes apply
  automatically when **Extract features Automatically** is ticked. Otherwise
  press **Extract features**.
- **LM Trainer** and **LoRA Trainer**: set the options, then press **Train**.
  **Stop** ends a run early and keeps what has been learned so far.
- **LM Generator**: press **Generate** after changing the prompt or sampling
  options.
- **Vocab Inspector** and **NN Viewer** are read-only; they update when an
  input changes.

## 6. Save your work

- Orange workflow: **File > Save As**.
- Keras language model: **Save model** in LM Trainer. This writes a directory
  containing the weights, vocabulary and metadata.
- LoRA adapter: **Save adapter** in LoRA Trainer. This writes a directory
  containing the adapter files and the base-model reference.

The Generator can load either directory with **Load model...**.

## 7. The LoRA adapter switch

When a LoRA model is connected, **LM Generator** enables a **Use LoRA adapter**
checkbox. Untick it to run the identical base model without the adapter. This
is the before/after comparison; no other setting changes.
