# 07 - LoRA supervised fine-tuning

Teach the model a question-answer pattern with prompt-response pairs, then
compare it with the raw base model.

## Before you start

Complete [06 - LoRA continued pre-training](06-lora-continued-pretraining.md)
or at least confirm that the converted model and `[lora]` extra are available.

The difference between this tutorial and the previous one is the *data format*,
not the loss function. Both train next-token cross-entropy. Here the rows are
authored prompt-response pairs, and mlx-lm masks the prompt tokens so the loss
is computed on the completion only.

## 1. Make a prompt-response Table

Save this as `data/text/pairs.txt`:

```text
Q: Who is the narrator of the story?
A: A woman spending the summer in an ancestral hall with her husband, John.

Q: What is John's profession?
A: He is a physician.

Q: What does John prescribe for the narrator?
A: Rest, air and tonics, and he forbids her to write.

Q: What room does the narrator stay in?
A: A nursery at the top of the house, with barred windows and yellow wallpaper.

Q: What does the narrator become obsessed with?
A: The pattern of the yellow wallpaper.

Q: What does she see behind the wallpaper?
A: A woman creeping and shaking the pattern.

Q: How does the narrator describe the wallpaper's smell?
A: It has a yellow smell that follows her through the house.

Q: What does she do on the last day?
A: She peels off the wallpaper and creeps around the room.

Q: What does John do when he sees her?
A: He faints.

Q: Who is the woman in the wallpaper?
A: The narrator herself, or the woman she imagines trapped behind it.
```

Add **LM Text**, set **Source → Type** to **File**, browse to
`data/text/pairs.txt`, switch **Table type** to **Q/A pairs**, and press
**Reload**. The preview table shows the parsed prompt and completion columns.

Connect **Data** to **LoRA Trainer** **Data**. The Table has `prompt` and
`completion` string metas; the source file stays in the preview only.

## 2. Run supervised fine-tuning

Set:

- **Data format**: Prompt-response pairs (SFT)
- **Base model**: absolute path to `data/models/britllm-4bit`
- **Iterations**: 20 for the first pass
- **Learning rate**: 1e-4
- **Fine-tuned layers**: 16
- **Batch size**: 1
- **Max seq length**: 128
- **Seed**: 42

The small batch is deliberate: the validation split must hold at least one full
batch. With 10 pairs and a 10% validation split, only 1 row is held out.

Press **Train**. The live loss curve shows the completion loss. Save the
adapter as `my-britllm-sft`.

## 3. Compare raw and SFT model

Connect *LoRA Trainer* **Model** → *LM Generator* **Model**.

Set:

- **Prompt text**:
  `Q: Who is the narrator of the story?` then a new line and `A:`
- **Temperature**: 0.0
- **Max new tokens**: 32
- **Number of generations**: 1

Generate with **Use LoRA adapter** ticked, then untick it and generate again.

With only 20 iterations the answer may not be correct in content, but the
*shape* of the response can change: it may adopt the answer pattern instead of
rambling. That behavioural change is the lesson. More pairs and more iterations
strengthen it.

## 4. Why this is supervised

A useful quiz: you have 500 sentences from the same domain. If you feed them to
LoRA as a raw corpus, is that supervised? No. Chunking text into sentences
changes document boundaries, not supervision.

Supervision lives in the pair format: a human chose one text as the prompt and
another as the desired completion. The loss is still next-token prediction, but
the prompt tokens are masked out, so capacity is spent on producing the
completion.

## 5. Try a variation

Add five more pairs in the same style to `data/text/pairs.txt`, press **Reload**
in LM Text, and retrain. Compare the same prompt. Watch whether the answer pattern becomes
more consistent.

## Checkpoint

- The Table has `prompt` and `completion` string columns.
- The LoRA Trainer accepts it as SFT data.
- The Generator shows the adapted and raw base models side by side.
- You can explain why the data format, not the loss function, makes this
  supervised.

## What next

- Analyse the generations with Orange's text widgets:
  [08 - Evaluation with Orange](08-evaluation.md)
- Revisit the sampling knobs:
  [05 - Generation and temperature](05-generation.md)
