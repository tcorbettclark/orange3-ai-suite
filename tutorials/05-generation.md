# 05 - Generation and temperature

Turn a trained model into visible text and find the temperature where it is
useful, repetitive, or nonsensical.

## 1. Connect a model

Use the LM Trainer from [04 - Train from scratch](04-train-from-scratch.md):

- *LM Trainer* **Model** → *LM Generator* **Model**

Alternatively press **Load model...** in the Generator and choose a saved
model directory.

## 2. First generation

Set:

- **Prompt text**: `the `
- **Temperature**: 0.2
- **Max new tokens**: 60
- **Number of generations**: 3
- **Seed**: 0

Press **Generate**. Connect **Generations** to **Data Table**. The output is a
one-column Table; the `generation` meta column contains the full text, prompt
included.

At low temperature the model is conservative. Expect repetition, common phrases
and a strong pull toward the most likely continuation.

## 3. Turn the temperature up

Try the same prompt at:

- 0.5 - still controlled, more varied;
- 1.0 - the model's natural sampling scale;
- 1.5 - surprising, sometimes ungrammatical;
- 2.5 - word salad.

Press **Generate** after each change. The lesson is not that one value is
"correct"; it is that sampling temperature changes the trade-off between
predictability and variety. For a tiny model, high temperature exposes how
little structure it has.

## 4. Seeds and determinism

Set **Temperature** to 0.0 (greedy). Generate once, then generate again with the
same seed. The texts should be identical.

Set **Seed** to `-1` and generate twice. The seed is now random, so the texts
differ.

This is the difference between training-time ground truth and generation-time
feedback: at each step the Generator feeds its own prediction back into the
model. A seed makes that loop reproducible.

## 5. Save and load a round trip

1. In LM Trainer, press **Save model** and save to a directory.
2. In LM Generator, press **Load model...** and select that directory.
3. Generate with temperature 0.0 and seed 0.

The loaded model should reproduce the same greedy text. This checks that the
weights *and* the tokenizer were saved together.

## 6. Compare two models

Train a bigram and an MLP on the same corpus. Connect each to its own Generator
(or swap the connection) and use the same prompt and temperature.

The bigram should sound locally plausible but lose the thread quickly. The MLP
should keep a short window of context. The comparison is the point: the
architecture's memory is visible in the text.

## 7. LoRA models

If you trained an adapter in [06 - LoRA continued pre-training](06-lora-continued-pretraining.md),
connect the LoRA Trainer **Model** output to the Generator. The **Use LoRA
adapter** checkbox becomes enabled:

- checked: the base model plus the adapter;
- unchecked: the identical base model without the adapter.

Generate the same prompt both ways. This is the clean before/after comparison;
only the adapter changes.

## Checkpoint

- Temperature changes repetition versus variety in a predictable direction.
- The same seed reproduces the same generation; `-1` randomises it.
- A saved model reproduces its greedy output after loading.
- LoRA models can be compared with the adapter on and off.

## What next

- Adapt a real base model:
  [06 - LoRA continued pre-training](06-lora-continued-pretraining.md)
- Supervised fine-tuning:
  [07 - LoRA supervised fine-tuning](07-lora-sft.md)
