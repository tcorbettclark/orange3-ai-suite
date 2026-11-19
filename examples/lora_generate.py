"""The other half of the boundary: load, stream and score, with the adapter off and on.

Run it from the repository root with
``uv run --extra lora python examples/lora_mlx_generate.py <model-dir> <adapter-dir>``,
where the adapter directory is the one ``examples/lora_mlx_run.py`` printed. The two
loads are the same base weights, so every difference below is the adapter's: the text
it produces, and the loss it gives to text it was, and was not, trained on.
"""

import argparse

from orangecontrib.ai_suite.kernel.lora import mlx_runtime

PROMPT = "Cirrus clouds are"
IN_DOMAIN = "Stratocumulus arrive as a layer of rounded rolls with gaps."
OUT_OF_DOMAIN = "The committee approved the budget after a short debate."

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("base", help="MLX model directory or Hugging Face repo id")
parser.add_argument("adapter", help="the adapter directory a run left behind")
args = parser.parse_args()

# The question comes before any import: `available` names the two modules without
# importing them, so a machine without MLX fails here, with the install line, rather
# than inside a Metal call.
if not mlx_runtime.available():
    raise SystemExit(
        "this needs the lora extra: pip install 'orange3-ai-suite[lora]' (macOS on Apple Silicon)"
    )


def sample(adapter_path: str | None) -> dict:
    """One prompt continued and two texts scored, on one variant of the model."""
    model, tokenizer = mlx_runtime.load_model(args.base, adapter_path)
    print(f"  loaded    model={type(model).__name__} adapter={adapter_path or 'off'}")

    # The chunks are mlx-lm's own slices, not words, and the completion arrives without
    # the prompt: join them for the text, and note that mlx-lm promises nothing about a
    # leading space. The `|` below only marks where the prompt ends.
    chunks = []
    print(f"  stream    {PROMPT}|", end="", flush=True)
    for chunk in mlx_runtime.stream_text(
        model, tokenizer, PROMPT, temperature=0.0, max_new_tokens=32, seed=0
    ):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print(f"\n            {len(chunks)} chunks from mlx-lm")

    scores = {
        "in-domain": mlx_runtime.evaluate_text(model, tokenizer, IN_DOMAIN, max_tokens=512),
        "out-of-domain": mlx_runtime.evaluate_text(model, tokenizer, OUT_OF_DOMAIN, max_tokens=512),
    }
    for where, score in scores.items():
        print(
            f"  score     {where:14s} loss {score['loss']:.3f}  "
            f"perplexity {score['perplexity']:.2f}"
        )
    return scores


print("base model")
before = sample(None)
print("\nwith the adapter")
after = sample(args.adapter)

print("\nthe gap, which is what the adapter is for")
for where in before:
    ratio = after[where]["perplexity"] / before[where]["perplexity"]
    print(
        f"  {where:14s} perplexity x{ratio:.3f}  "
        f"({before[where]['perplexity']:.2f} -> {after[where]['perplexity']:.2f})"
    )
print("  a fall on the in-domain text is the fit; the out-of-domain line is the cost.")
