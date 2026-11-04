# 08 - Evaluation with Orange

Turn generations into corpora and inspect them with Orange's text widgets. This
tutorial is optional: install Orange3-Text to unlock the analysis widgets.

## 1. Install Orange3-Text

In the canvas, open **Options > Add-ons**, find **Orange3-Text** and install it.
Restart Orange.

## 2. Generate comparable outputs

Use [05 - Generation and temperature](05-generation.md) to produce two sets of
generations:

- the same prompt at two temperatures; or
- the same prompt with a LoRA adapter on and off; or
- the same prompt from a bigram and an MLP.

Keep the prompt, seed and maximum new tokens identical so the comparison is
fair.

## 3. Table to corpus

Connect *LM Generator* **Generations** to **Corpus** **Data**.

The Corpus widget turns the text column into documents. Its **Corpus** output
can feed:

- **Word Cloud** - the vocabulary the model actually produces;
- **Concordance** - the contexts around a chosen word;
- **Document Map** - where the generated documents sit relative to each other;
- **Document Embedding** - the vectors behind the map.

Use **Word Cloud** first. Select a word and send it to **Concordance**. Read the
surrounding text: does the model use the word in a plausible context, or does it
produce a collocation that never occurs in the source corpus?

That is the hallucination fingerprint the evaluation is looking for.

## 4. Compare vocabularies

Generate from the raw base model and from the adapted model (adapter off/on in
the Generator). Send each set through its own Corpus and Word Cloud.

Questions to ask:

1. Which model uses more domain words?
2. Which model repeats the prompt's words?
3. Which model produces words that are absent from the source corpus?
4. Does higher temperature increase vocabulary variety, or just noise?

## 5. Compare with the source corpus

Load the same downloaded story with **LM Text** and connect its **Data** output
through **Corpus** to **Word Cloud**. Inspect that cloud next to the generated
one. The source is the reference for "domain-plausible". A word that appears in
the generated cloud but not in the source is a candidate hallucination or a
sampling artefact.

## 6. Read the raw text

For a direct side-by-side, connect **Generations** to **Data Table** or **Corpus
Viewer**. The prompt is included in each generation, so you can see whether the
model continued the prompt, answered it, or ignored it.

## Checkpoint

- You can move a generation Table into a Corpus.
- You can read a Word Cloud and a Concordance for the same output.
- You can state one difference between raw and adapted model outputs, and one
  difference between low and high temperature.

## What next

These tutorials are also manual test scripts. When a widget behaves in a way
that makes a step awkward, that is a concrete improvement to make: a missing
control, an unclear error, a default that should be different, or a comparison
that needs to be easier to set up.
