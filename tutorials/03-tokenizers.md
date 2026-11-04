# 03 - Tokenizers

Look inside the vocabulary before training anything. The tokenizer decides what
a model can see, so it is the first thing to make concrete.

## 1. Load a text Table

Add **LM Text Loader** and load `data/text/yellow-wallpaper.txt` as described in
[00 - Setup](00-setup.md#3-load-text-with-lm-text-loader). Use **Paragraphs** and
enable the Gutenberg and reflow cleaning rules.

Connect its **Data** output to **LM Vocab Viewer** **Data**.

## 2. Compare character and word level

Open **LM Vocab Viewer**. The summary line reports vocabulary size, mean token
length and the fraction of tokens seen only once (the hapax fraction).

- **Char level** - every character is a token, spaces included. The vocabulary
  is small; the sequences are long.
- **Word level** - words and punctuation marks are tokens. The vocabulary is
  larger; the sequences are shorter.

Read the **Char level** and **Word level** tabs. Then answer:

1. Which level has more tokens seen exactly once?
2. Which level has the longest individual tokens?
3. What happens to punctuation at word level? Look for `.`, `,` and `"` in the
   token list.
4. What happens to capital letters? Case is preserved; `The` and `the` are
   different tokens.

## 3. Change the text

Create a second `.txt` file with an apostrophe and a quotation:

```text
The king's counsellors said, "We will not wait."
The physician's satchel held herbs, a knife and a tincture.
```

Load it with **LM Text Loader** and press **Reload** and inspect again. At word level, `king's` should stay one token;
quote marks are separate tokens. This is the dependency-free tokenizer the
kernel uses, not a pretrained subword tokenizer.

## 4. Inspect a trained model

Train a model in [04 - Train from scratch](04-train-from-scratch.md). Then
connect its **Model** output to **LM Vocab Viewer** **Model**.

The **Model tokenizer** tab shows the vocabulary that travelled with the model.
This is the important invariant: the embedding row that means one token only
means that token because this vocabulary says so.

If both a model and data are connected, the model wins and a warning says the
data input is ignored. The viewer also has no outputs: it cannot feed a
mismatched vocabulary back into training or generation.

## 5. Test the hypothesis

The LM Pre-trainer's tokenizer choice is a testable hypothesis:

- char level should give a smaller vocabulary and longer sequences;
- word level should give a larger vocabulary and shorter sequences, but more
  `<unk>` tokens on unfamiliar text.

Compare the numbers in **LM Vocab Viewer** with the loss curves in the LM Pre-trainer.
Write down which level trained faster and which produced more plausible text.

## Checkpoint

- You can read vocabulary size, mean token length and hapax fraction.
- You can explain why the tokenizer travels with the model.
- You can predict what a word-level model will do with an unseen word.

## What next

- Train the three architectures:
  [04 - Train from scratch](04-train-from-scratch.md)
