"""Use a trained adapter: the same weights with it off and on, then saved and reloaded.

Run it from the repository root with ``uv run --extra lora python
examples/lora_generate.py <artifact-dir>``, where the artifact is what
``examples/lora_train.py --save`` wrote. One ``LoRAModel`` holds both variants, so the
only thing that changes between the two halves is ``adapter_enabled`` - and the
artifact names its own base model, so that is all the path this script needs.
"""

import argparse
import tempfile
from pathlib import Path

import orangecontrib.ai_suite.kernel.lora as lora

PROMPT = "Cirrus clouds are"
IN_DOMAIN = "Stratocumulus arrive as a layer of rounded rolls with gaps."
OUT_OF_DOMAIN = "The committee approved the budget after a short debate."

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact", help="the artifact directory lora_train.py --save wrote")
args = parser.parse_args()

if not lora.available():
    raise SystemExit(
        "this needs the lora extra: pip install 'orange3-ai-suite[lora]' (macOS on Apple Silicon)"
    )

# The artifact carries the base model's identity, the adapter files and the run's
# record; loading it is what turns those three into a model. The weights themselves
# are read on first use, one variant at a time.
model = lora.LoRAModel.from_artifact(args.artifact)
record = model.training
print(f"artifact    {model.name or '(unnamed)'} fitted to {Path(model.base_model).name}")
print(
    f"            {record['iters_completed']} of {record['options']['iters']} iterations,"
    f" stopped={record['stopped']}"
)


def show(with_adapter: bool) -> dict:
    """One prompt continued and two texts scored, on one variant of the model."""
    model.adapter_enabled = with_adapter
    print(f"\nadapter     {'on' if with_adapter else 'off'}")

    # `generate` is `stream` joined up, which is the kernel's convention for every
    # model in the suite: the chunks are the library's, so never depend on where they
    # fall. Greedy here, so the same prompt gives the same text twice.
    text = model.generate(PROMPT, temperature=0.0, max_new_tokens=32, seed=0)
    pieces = list(model.stream(PROMPT, temperature=0.0, max_new_tokens=32, seed=0))
    print(f"  text      {text}")
    print(
        f"  stream    {len(pieces)} chunks, joined back to the same text: {''.join(pieces) == text}"
    )

    scores = {
        "in-domain": model.evaluate(IN_DOMAIN),
        "out-of-domain": model.evaluate(OUT_OF_DOMAIN),
    }
    for where, score in scores.items():
        print(
            f"  score     {where:14s} loss {score['loss']:.3f}"
            f"  perplexity {score['perplexity']:.2f}"
        )
    return scores


before = show(False)
after = show(True)

print("\nthe gap, which is what the adapter is for")
for where in before:
    ratio = after[where]["perplexity"] / before[where]["perplexity"]
    print(
        f"  {where:14s} perplexity x{ratio:.3f}"
        f"  ({before[where]['perplexity']:.2f} -> {after[where]['perplexity']:.2f})"
    )
print("  a fall on the in-domain text is the fit; the out-of-domain line is the cost.")

# Saving is copying the adapter out of the directory the run owned, plus the record:
# the artifact is then a directory that can be moved, shared and loaded without it.
with tempfile.TemporaryDirectory(prefix="lora-saved-") as scratch:
    saved = model.save(Path(scratch) / "artifact")
    reloaded = lora.LoRAModel.from_artifact(saved)
    print(
        f"\nreloaded    name={reloaded.name!r} adapter={reloaded.adapter_enabled}"
        f" iterations={reloaded.training['iters_completed']}"
    )
