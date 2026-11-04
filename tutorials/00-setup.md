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

The committed demo data is enough for the image tutorials:

- `demo/mnist/train/` - PNG digits in one subfolder per label.

The cloud corpus and the BritLLM model are optional. Fetch them when a tutorial
asks for them:

```bash
uv run python demo/fetch_clouds.py
uv run --extra lora python demo/fetch_britllm.py
```

For the language-model tutorials, download a public-domain short story. The
tutorials use Charlotte Perkins Gilman's *The Yellow Wallpaper*, but any UTF-8
`.txt` file works:

```bash
mkdir -p data/text
curl -L https://www.gutenberg.org/ebooks/1952.txt.utf-8 \
    -o data/text/yellow-wallpaper.txt
```

## 3. Load text with LM Text

The LM widgets accept a Table with string text columns. Add **LM Text** to the
canvas:

1. Set **Source → Type** to **File**.
2. **Browse...** to `data/text/yellow-wallpaper.txt`.
3. Press **Reload**.
4. Leave **Table type** as **Text documents** and **Split into** as
   **Paragraphs**.
5. Enable **Strip Project Gutenberg header/footer** and **Reflow wrapped
   lines** in the cleaning section.
6. Connect **Data** to **LM Trainer**, **Vocab Inspector** or **LoRA Trainer**.

The preview table shows the first rows. Select one to read its full text below.
Changing split or cleaning options reprocesses the cached file; **Reload**
rereads it from disk.

For a folder, switch **Type** to **Folder** and choose a directory. LM Text
recurses through it, reads every `.txt`, and keeps a `source` column.

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
button and point it at `demo/mnist/train` or `data/clouds`.

If you prefer to build from scratch, every tutorial lists the widgets and
links.

## 5. Controls that apply automatically, and controls that do not

- **Image Feature Extractor**: preprocessing and group changes apply
  automatically. Type a new working size and press Enter to reprocess.
- **LM Text**: press **Reload** to reread the file or folder; changing the
  table type, split mode or cleaning options reprocesses the cached text.
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
