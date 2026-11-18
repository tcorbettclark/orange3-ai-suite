# 00 - Setup

Get the add-on, the data and a text Table ready. Everything else in the
tutorials builds on this page.

## 1. Install and launch

Follow the project [README](../README.md#installation) to install the add-on
into the Orange interpreter. For the LoRA tutorials you also need the `[lora]`
extra in that interpreter:

```bash
uv pip install --python /Applications/Orange.app/Contents/MacOS/python \
    "orange3-ai-suite[lora]"
```

The LoRA tutorials also need the 4-bit BritLLM model. Add **BritLLM Models** to
the canvas and press **Download** (1.4 GB); the model is cached in the Hugging
Face hub cache (`HF_HOME`, default `~/.cache/huggingface`). Connect its
**Model path** output to **Base model** on LoRA Pre-trainer or LoRA SFT.

Keras, TensorFlow and OpenCV are plain dependencies, so one sync is enough:

```bash
uv sync
```

Launch the canvas from the repository root:

```bash
/Applications/Orange.app/Contents/MacOS/python -m Orange.canvas
```

The widgets are in three categories: **Language Models**, **Image Analytics**
and **Visualize**.

## 2. Data

The image tutorials use two pinned Hugging Face assets; no `demo/` or `data/`
checkout is needed:

- **MNIST digits** - `orange-ai-assets fetch mnist-digits` (or
  `get_mnist()`). The prepared root has `train/` and `val/`, 28x28, 500/100
  images per digit.
- **CCSN clouds** - `orange-ai-assets fetch ccsn-clouds` (or `get_clouds()`).
  The prepared root has `ac/`, `sc/`, `cu/`, `cb/`, `ci/` and `CREDITS.md`.
  The CCSN database is CC0; cite Zhang et al. (2018), *CloudNet*.

Both commands print the prepared root; use it for **Import Images**. The
BritLLM model is downloaded by the **BritLLM Models** widget (or headlessly
with `orange-ai-assets fetch britllm-4bit`); the command prints
the snapshot path to paste into **Base model**.

For the language-model tutorials, download a public-domain short story. The
tutorials use Charlotte Perkins Gilman's *The Yellow Wallpaper*, but any UTF-8
`.txt` file works:

```bash
mkdir -p data/text
curl -L https://www.gutenberg.org/ebooks/1952.txt.utf-8 \
    -o data/text/yellow-wallpaper.txt
```

## 3. Load text with LM Text Loader

The LM widgets accept a Table with string text columns. Add **LM Text Loader** to the
canvas:

1. Set **Source → Type** to **File**.
2. **Browse...** to `data/text/yellow-wallpaper.txt`.
3. Press **Reload**.
4. Leave **Table type** as **Text documents** and **Split into** as
   **Paragraph**.
5. Enable **Strip Project Gutenberg header/footer** and **Reflow wrapped
   lines** in the cleaning section.
6. Connect **Data** to **LM Pre-trainer**, **LoRA Pre-trainer**, **LoRA SFT** or
**LM Vocab Viewer**.

The preview table shows the first rows. Select one to read its full text below.
Changing split or cleaning options reprocesses the cached file; **Reload**
rereads it from disk.

For a folder, switch **Type** to **Folder** and choose a directory. LM Text Loader
recurses through it and reads every `.txt`. The preview shows each row's source
file, but the Data output contains only the text.

For Q/A experiments, switch **Table type** to **Q/A pairs**. The fixed format is
shown in the widget:

```text
Q: Who is the narrator?
A: A woman spending the summer in an ancestral hall.

Q: What does she become obsessed with?
A: The yellow wallpaper in her room.
```

## 4. Open a prepared workflow

The repository ships two canvases:

- `demo/mnist_features.ows`
- `demo/cloud_features.ows`

Use **File > Open** and choose one. The Import Images paths inside are
machine-local. If a path is stale, open **Import Images**, click the folder
button and point it at the prepared MNIST `train/` or CCSN root printed by
`orange-ai-assets fetch mnist-digits` / `orange-ai-assets fetch ccsn-clouds`.

If you prefer to build from scratch, every tutorial lists the widgets and
links.

## 5. Controls that apply automatically, and controls that do not

- **Image Features**: preprocessing and group changes apply
  automatically. Type a new working size and press Enter to reprocess.
- **LM Text Loader**: press **Reload** to reread the file or folder; changing the
  table type, split mode or cleaning options reprocesses the cached text.
- **LM Pre-trainer**, **LoRA Pre-trainer** and **LoRA SFT**: set the options,
  then press **Train**.
  **Stop** ends a run early and keeps what has been learned so far.
- **LM Generator**: press **Generate** after changing the prompt or sampling
  options.
- **LM Vocab Viewer** and **Neural Network Viewer** are read-only; they update when an
  input changes.

## 6. Save your work

- Orange workflow: **File > Save As**.
- Trained models currently travel through the canvas as **Model** signals.
  Persistence to disk is being moved to dedicated Load LM/Save LM widgets, so
  the training and generation widgets do not have their own save/load buttons.

## 7. The LoRA adapter switch

When a LoRA model is connected, **LM Generator** enables a **Use LoRA adapter**
checkbox. Untick it to run the identical base model without the adapter. This
is the before/after comparison; no other setting changes.
