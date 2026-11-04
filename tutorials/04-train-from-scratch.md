# 04 - Train from scratch

Train three tiny language models on the same corpus and compare what each one
can remember. This is the "from scratch" end of the language-model arc.

## 1. Load the training Table

Add **LM Text Loader** and load `data/text/yellow-wallpaper.txt` as described in
[00 - Setup](00-setup.md#3-load-text-with-lm-text-loader). Use **Paragraphs**, and
enable the Gutenberg and reflow cleaning rules.

Connect its **Data** output to **LM Pre-trainer** **Data**.

If you want a faster first pass, choose a shorter `.txt` file or **Lines**
instead. The steps are identical; only the corpus size changes.

## 2. Bigram: the floor

Set:

- **Architecture**: Bigram
- **Tokenizer**: Char level
- **Epochs**: 5
- **Batch size**: 256
- **Sample prompt**: `the `
- **Sample temperature**: 0.0
- **Sample length**: 40

Press **Train**. Watch the live loss curve and the status line. The bigram
predicts the next token from one token of context. Its loss falls quickly and
then plateaus; its samples are pronounceable garbage.

Connect **Checkpoint** to **Data Table**. There is one row per epoch. Read the
`sample` column from top to bottom. This is the checkpoint table: the text
evolving as the model learns.

## 3. MLP over a window: the default

Set:

- **Architecture**: MLP-over-window
- **Context window**: 16
- **Embedding dim**: 32
- **Hidden units**: 128
- **Epochs**: 8
- **Batch size**: 256
- **Sample prompt**: `the `
- **Sample temperature**: 0.0
- **Sample length**: 60

Press **Train**. Compare three things with the bigram:

1. The loss curve falls further.
2. The checkpoint samples move from gibberish to period pastiche.
3. The model has a hard, visible memory: it sees exactly the last 16 tokens.

The status line after training reports the number of training and validation
windows and the final validation loss and perplexity. Perplexity is `e^loss`:
roughly, "the model is choosing among this many plausible next tokens".

## 4. LSTM: recurrent memory

Set:

- **Architecture**: LSTM
- **Context window**: 32
- **Embedding dim**: 64
- **LSTM units**: 128
- **LSTM layers**: 1 layer
- **Epochs**: 2
- **Batch size**: 128

Press **Train**. The LSTM carries a compressed hidden state instead of a fixed
window. Its curve is usually smoother, and it costs more per epoch. Treat it as
a launch-and-return experiment: start it, then inspect the samples when it
finishes.

## 5. Character level versus word level

Repeat the MLP run with:

- **Tokenizer**: Word level
- **Max vocab**: 5000
- **Context window**: 8

Look at the vocabulary in **LM Vocab Viewer** (connect the model's **Model**
output to its **Model** input). Word level gives a larger vocabulary and shorter
sequences. Compare the loss curve and the samples with the char-level run.

## 6. Stop early

Set **Epochs** to 100 and press **Train**. After a few epochs press **Stop**.
The run ends cleanly and the partially trained model is still emitted. Check
the checkpoint table and generate from it in
[05 - Generation and temperature](05-generation.md).

## 7. Keep the model in the workflow

The trained model is emitted on the **Model** output and can be connected
directly to **LM Generator**. The LM Pre-trainer no longer has a Save button;
persistence will return as a dedicated Save LM widget.

## Checkpoint

- The bigram plateaus; the MLP learns longer context; the LSTM has recurrent
  state.
- The checkpoint table shows the text evolving epoch by epoch.
- You can save a model directory and explain why the tokenizer is inside it.

## What next

- Generate text and turn the temperature knob:
  [05 - Generation and temperature](05-generation.md)
- Inspect the vocabulary:
  [03 - Tokenizers](03-tokenizers.md)
