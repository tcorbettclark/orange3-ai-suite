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

Add **Python Script** and paste:

```python
import numpy as np
from Orange.data import Domain, StringVariable, Table

pairs = [
    ("Q: In which century did the empire of Rome reach its greatest extent?\nA:",
     "In the second century of the Christian era."),
    ("Q: Who were the emperors named by Gibbon in the age of the Antonines?\nA:",
     "Nerva, Trajan, Hadrian, and the two Antonines."),
    ("Q: What guarded the extensive monarchy of Rome?\nA:",
     "Ancient renown and disciplined valor."),
    ("Q: What influence cemented the union of the provinces?\nA:",
     "The gentle but powerful influence of laws and manners."),
    ("Q: What did the frontiers of the empire enjoy?\nA:",
     "The advantages of wealth and luxury."),
    ("Q: Who acquired the principal conquests of the Romans?\nA:",
     "The Romans achieved them under the republic."),
    ("Q: Which city was the seat of government in Britain?\nA:",
     "York."),
    ("Q: What did Gaul boast of under the empire?\nA:",
     "Her twelve hundred cities."),
    ("Q: What was the condition of the imperial constitution?\nA:",
     "The image of a free constitution was preserved with decent reverence."),
    ("Q: What happened after the death of Marcus Antoninus?\nA:",
     "The empire entered its decline and fall."),
]

pairs = pairs * 2  # 20 rows; enough to hold a small validation split

domain = Domain(
    [],
    metas=[StringVariable("prompt"), StringVariable("completion")],
)
out_data = Table.from_numpy(
    domain,
    np.empty((len(pairs), 0)),
    None,
    np.asarray(pairs, dtype=object).reshape(-1, 2),
)
```

Press **Run**. Connect **Data** to **LoRA Trainer** **Data**.

The Table has two string metas named `prompt` and `completion`. The adapter
looks for those names first; if they are absent, it takes the first two string
columns.

## 2. Run supervised fine-tuning

Set:

- **Data format**: Prompt-response pairs (SFT)
- **Base model**: absolute path to `data/models/britllm-4bit`
- **Iterations**: 20 for the first pass
- **Learning rate**: 1e-4
- **Fine-tuned layers**: 16
- **Batch size**: 2
- **Max seq length**: 128
- **Seed**: 42

The small batch is deliberate: the validation split must hold at least one full
batch. With 20 pairs and a 10% validation split, only 2 rows are held out.

Press **Train**. The live loss curve shows the completion loss. Save the
adapter as `my-britllm-sft`.

## 3. Compare raw and SFT model

Connect *LoRA Trainer* **Model** → *LM Generator* **Model**.

Set:

- **Prompt text**:
  `Q: Who wrote The History of the Decline and Fall of the Roman Empire?` then
  a new line and `A:`
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

Add five more pairs in the same style, or repeat the existing pairs more times.
Retrain and compare the same prompt. Watch whether the answer pattern becomes
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
