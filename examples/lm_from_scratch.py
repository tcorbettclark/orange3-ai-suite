"""Train a tiny language model from scratch, then generate and score text with it.

Run it from the repository root:

    uv run python examples/lm_from_scratch.py

The four objects, and the lines where they meet:

    text ──tokenizer_from_text──> Tokenizer
    MLP(...) ───────────────────> Architecture
    LanguageModel(architecture, tokenizer)      # nothing trained yet
        .fit(text, TrainingOptions(...))        # pairs, split, graph, weights
        .generate(prompt, temperature)          # context -> logits -> id
        .evaluate(held_out_text)                # loss and perplexity
"""

import math

from orangecontrib.ai_suite.kernel.lm import (
    MLP,
    LanguageModel,
    TrainingOptions,
    tokenizer_from_text,
)

CORPUS = """the cat sat on the mat.
the cat saw the dog.
the dog ran to the cat.
the cat ran to the mat.
a dog and a cat sat on the mat.
the mat sat on the dog."""

# 1. A tokenizer owns the vocabulary and the ids; it travels with the model.
tokenizer = tokenizer_from_text(CORPUS, "char")
print(
    f"tokenizer   name={tokenizer.tokenizer_name()!r} vocab={tokenizer.vocab_size} "
    f"corpus tokens={tokenizer.total_token_count}"
)
print(f"            encode('the cat') -> {tokenizer.encode('the cat')}")
print(f"            decode back       -> {tokenizer.decode(tokenizer.encode('the cat.'))!r}")

# 2. An architecture owns the graph shape and its own hyperparameters. No weights.
architecture = MLP(context_width=16, embedding_dim=16, hidden_units=64)
print(
    f"\narchitecture {architecture.architecture_name()!r} "
    f"context_width={architecture.context_width} params={architecture.get_params()}"
)

# 3. A model is an architecture plus a tokenizer. Nothing is trained yet.
model = LanguageModel(architecture, tokenizer)
print(f"model       keras_model={model.keras_model}  results={model.results}")

# 4. fit() builds pairs at architecture.context_width, splits them by unit,
#    builds and compiles the Keras model, trains it, and returns the same object.
epochs = []


def on_epoch(stats):
    """One dict per epoch - the numbers the widget plots - plus our own sample."""
    epochs.append(stats)
    stats["sample"] = model.generate("the cat ", temperature=0.5, max_new_tokens=40)


model.fit(CORPUS, TrainingOptions(epochs=40, batch_size=16, seed=0), on_epoch=on_epoch)

print(
    f"\nfit         keras_model={type(model.keras_model).__name__} "
    f"input_shape={model.keras_model.input_shape}  "
    f"(None, {architecture.context_width})"
)
print(
    f"            pairs: {model.results.n_pairs_total} total, "
    f"{model.results.n_train_pairs} train, {model.results.n_val_pairs} val"
)
print(
    f"            loss {model.results.final_loss:.3f} "
    f"(perplexity {model.results.final_perplexity:.2f}), "
    f"val {model.results.final_val_loss:.3f} "
    f"(perplexity {model.results.final_val_perplexity:.2f})"
)
print(f"            first epoch {epochs[0]['loss']:.3f} -> last epoch {epochs[-1]['loss']:.3f}")
print(f"            epoch stats keys: {sorted(epochs[0])}")
print(f"            sample at epoch 20: {epochs[19]['sample']!r}")

# 5. generate() samples one unit, sliding a context of context_width ids and
#    stopping when the model emits EOS.
print("\ngenerate    'the cat ', temperature 0.5")
print("           ", model.generate("the cat ", temperature=0.5, max_new_tokens=60, seed=0))
print("generate    '', temperature 0 (greedy)")
print("           ", model.generate("", temperature=0, max_new_tokens=60))

# 6. evaluate() scores the fitted weights on text they were not trained on:
#    in-domain perplexity should be far below out-of-domain.
in_domain = model.evaluate("the cat sat on the mat.")
out_domain = model.evaluate("quantum chromodynamics is a theory.")
print(
    f"\nevaluate    in-domain  loss {in_domain['loss']:.2f} "
    f"perplexity {in_domain['perplexity']:.2f}"
)
print(
    f"            out-domain loss {out_domain['loss']:.2f} "
    f"perplexity {out_domain['perplexity']:.2f}"
)
print(f"            check: e^loss = {math.exp(in_domain['loss']):.2f}")
