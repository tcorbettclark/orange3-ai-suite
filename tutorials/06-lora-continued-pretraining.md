# 06 - LoRA continued pre-training

Adapt a real pretrained model to a public-domain story with 4-bit QLoRA, then
compare the base model and the adapted model on identical prompts.

## Before you start

You need:

- macOS on Apple Silicon;
- the `[lora]` extra installed in the Orange interpreter;
- a converted 4-bit model at `data/models/britllm-4bit/`.

The setup tutorial has the install and conversion commands:
[00 - Setup](00-setup.md#1-install-and-launch).

Get the absolute model path:

```bash
python -c "from pathlib import Path; print(Path('data/models/britllm-4bit').resolve())"
```

Paste that path into the LoRA Trainer's **Base model** field. A Hugging Face id
is treated as unquantised and is rejected; the Trainer only accepts a verifiably
4-bit model.

## 1. Load the corpus Table

Add **LM Text** and load `data/text/yellow-wallpaper.txt` as described in
[00 - Setup](00-setup.md#3-load-text-with-lm-text). Use **Paragraphs**, and
enable the Gutenberg and reflow cleaning rules.

Connect **Data** to **LoRA Trainer** **Data**.

## 2. Continued pre-training

Set:

- **Data format**: Raw corpus (continued pre-training)
- **Iterations**: 30 for the first pass
- **Learning rate**: 1e-4
- **Fine-tuned layers**: 16
- **Batch size**: 4
- **Max seq length**: 512
- **Seed**: 42

Press **Train**. The live plot shows loss per iteration. The status line shows
the current iteration, loss and tokens per second.

What is happening:

- the base weights are frozen;
- small low-rank adapter matrices are trained;
- the loss is still next-token cross-entropy, so this is self-supervised
  continued pre-training, not instruction tuning.

The model output is a `LoRAModel` holding the base model and the adapter. The
adapter is not permanent until you press **Save adapter**; do that now, for
example as `my-britllm-cpt`.

If training reports that the validation split is smaller than the batch size,
reduce **Batch size** or provide more data.

## 3. Compare base and adapted model

Connect *LoRA Trainer* **Model** → *LM Generator* **Model**.

Set:

- **Prompt text**: `It is very seldom that mere ordinary people like John and myself`
- **Temperature**: 0.0
- **Max new tokens**: 48
- **Number of generations**: 1

Generate with **Use LoRA adapter** ticked. Then untick it and generate again.

Both outputs come from the identical base weights; only the adapter differs.
At 30 iterations the difference may be subtle. Increase **Iterations** to 100 or
300 for a stronger domain effect. The point of the first run is to see the
mechanism, not to claim convergence.

## 4. A second prompt

Try a prompt that a raw base model will continue rather than answer:

```text
Q: Who wrote The Yellow Wallpaper?
A:
```

The raw base model is a base model: it has never been instruction-tuned. It may
continue the pattern of the prompt without answering the question. This is the
deficiency that the SFT tutorial fixes.

## 5. Read the number that matters

The status/output area reports the trainable-parameter share: the adapter
moves a small fraction of the base model's parameters. That is the efficiency
lesson. You are not retraining the model; you are adding a small, removable
correction.

## Checkpoint

- LoRA training runs in the background and streams loss.
- The adapter and its base-model reference are one artifact.
- The Generator can run the same model with the adapter on or off.
- A raw base model continues text; it does not automatically follow
  instructions.

## What next

- Teach it to follow instructions:
  [07 - LoRA supervised fine-tuning](07-lora-sft.md)
- Return to sampling:
  [05 - Generation and temperature](05-generation.md)
