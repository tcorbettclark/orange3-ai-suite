# Tutorials

Hands-on walk-throughs for the Neural Networks add-on. Each tutorial builds an
Orange canvas and ends with something you can inspect, compare or reuse. They
work as classroom handouts, as manual test scripts, and as starting points for
extending the widgets.

## Before you start

1. Install the add-on into the Orange environment and launch the canvas. The
   commands are in the project [README](../README.md#installation).
2. Work from the repository root, so the relative paths in these tutorials
   point at the bundled demo data.
3. The MNIST and Gibbon data are committed in `demo/`. The cloud corpus and the
   BritLLM model are optional and are fetched by scripts in `demo/`.

```bash
uv run --extra lm python demo/fetch_mnist.py   # only needed to rebuild demo/mnist
uv run python demo/fetch_gibbon.py     # only needed to rebuild the corpus
uv run python demo/fetch_clouds.py     # optional cloud-corpus tutorial
uv run --extra lora python demo/fetch_britllm.py   # optional LoRA tutorials
```

## Conventions

- **Widget**: an Orange box, for example *LM Trainer*.
- **Link**: an arrow between an output and an input, for example *Model* to
  *Model*.
- **Run**: some widgets compute when a button is pressed; others apply
  automatically. The text says which.
- *Python Script* is a standard Orange widget. Paste the snippets from
  [00-setup.md](00-setup.md), press **Run**, and connect its **Data** output.
  This is only used to make text Tables without an extra add-on.

## The tutorials

| Tutorial | Builds | Needs |
|---|---|---|
| [00 - Setup](00-setup.md) | Installation, data, text Tables | add-on |
| [01 - Image features](01-image-features.md) | Import Images → features → model → evaluation | MNIST |
| [02 - Colour shortcut and baselines](02-colour-shortcut.md) | Cloud workflow, group ablation, several learners | cloud corpus |
| [03 - Tokenizers](03-tokenizers.md) | Data → Vocab Inspector | text Table |
| [04 - Train from scratch](04-train-from-scratch.md) | Bigram, MLP-over-window and LSTM training | text Table |
| [05 - Generation and temperature](05-generation.md) | LM Trainer → LM Generator, save/load | tutorial 04 |
| [06 - LoRA continued pre-training](06-lora-continued-pretraining.md) | Base model → LoRA Trainer → Generator | Apple Silicon, BritLLM |
| [07 - LoRA supervised fine-tuning](07-lora-sft.md) | Prompt-response pairs, prompt masking | Apple Silicon, BritLLM |
| [08 - Evaluation with Orange](08-evaluation.md) | Generations → text-analysis widgets | Orange3-Text (optional) |

## A note on the two teaching streams

The image tutorials use hand-crafted features and Orange's ordinary learners.
The language-model tutorials move from a model trained from scratch, to a real
pretrained base model, to an adapted model. Both streams end in Orange's
evaluation widgets, so the comparison is visual rather than asserted.
