# Glossary

## Processes

- **tokenizer**: `text` → `stream` — the whole corpus as one list of ids.
- **encode**: `text` → `stream` — EOS closes every unit, and anything the
  vocabulary cannot hold becomes `<unk>`.
- **decode**: `stream` → `text` — the reverse; BOS and EOS are dropped, `<unk>`
  stays literal.
- **pair builder**: `stream` → `pairs` — one context and one target per prediction.
- **fit**: `architecture` + `text` → a `model` — build the pairs, then the weights
  that minimise the loss.
- **model**: `context` → `logits` — one score per token in the vocabulary.
- **sampler**: `logits` → `id` — one id per step of a generation.
- **evaluate**: `model` + `text` → loss and perplexity, on any text. Report it on
  text the weights were not trained on to see the in-domain / out-of-domain gap.
- **save** / **load**: `model` ↔ a directory — the weights (`model.keras`), the
  tokenizer, and the run's settings and results (`meta.json`).

## Text

- **text** — a string a person wrote.
- **corpus** — all the text being trained on.
- **unit** — one self-contained piece of text. Currently a unit is a sentence.

## Tokens

Tokens are built by the tokenizer (`kernel/lm/tokenizer.py`).

- **token** — the smallest piece the tokenizer emits: one character (char
  tokenizer) or one word or punctuation mark (word tokenizer).
- **id** — a token's integer. Opaque: only the tokenizer's table gives it
  meaning, and ids never mean anything across two tokenizers.
- **vocabulary** — the set of tokens the tokenizer knows. **`vocab_size`** — how
  many, which is also how many logits the model emits.
- **stream** — the corpus as one flat list of ids. Each unit ends with EOS. The
  stream is the only input to the pair builder.
- **special token** — one of BOS, `<unk>` or EOS: the tokens every vocabulary
  begins with, and the only ones with no corpus count. "The specials" for short.
- **EOS** — the id that closes a unit. It travels inside the stream, and it is
  itself a target.
- **BOS** — the id meaning "nothing came before". It never travels in the
  stream: the pair builder opens a unit's context with it, and pads a context
  out with it when the unit has fewer than `w` ids before the target.
- **count** — how often a real token occurs in the corpus the vocabulary was
  built from. Zero for the specials, which are not counted.
- **rank** — a real token's position when tokens are ordered by count, the most
  frequent first; equal counts are ordered by spelling. Vocabulary order is not
  rank: it carries no meaning at all.
- **frequency table** — the top `n` real tokens as (rank, token, count). The Zipf
  summary is built from it; the vocabulary viewer shows the whole rank order.
- **total token count** — the corpus tokens counted when the vocabulary was
  built. Pruning leaves it alone, so shares stay corpus-relative rather than
  being measured against the survivors.
- **pruning** — removing real tokens from a vocabulary, by `keep_top(n)` or by a
  frequency floor `drop_rare(min_count=…)`. The specials always survive, and the
  remaining counts and the total are preserved.
- **max vocab** — the widget's cap on the vocabulary, applied by `keep_top`
  before training; `0` keeps everything.
- **dropped occurrences** — corpus occurrences that pruning made
  unrepresentable. They are the difference between the total token count and
  the sum of the counts, and they encode as `<unk>`.
- **`<unk>`** — the id for anything the vocabulary cannot represent: a token
  never seen when the vocabulary was built, or a real token that pruning
  removed. It appears in the stream like any other id, so it can be a **target**
  as well as part of a context, and a model can learn to emit it.

## Context

- **context** — the ids the model reads to make one prediction. At most `w` of
  them are ids of the text; the rest are BOS, because the unit had nothing
  before the target. Left-padded to width `w`, always.
- **slot** — one place in a context, holding one id. Slots are ordered: the
  **rightmost slot holds the id immediately before the target**, and moving
  left goes further back.
- **context width `w`** — how many slots a context has: how far back the model
  reads. A hyperparameter of `MLP` and `LSTM`, enumerable through `get_params()`,
  and the constant 1 for `Bigram`, which has no such parameter: a model that
  reads more history is an n-gram of higher order, a different architecture
  rather than a setting.

## Pairs

Pairs are built by the pair builder (`kernel/lm/pairs.py`).

- **target** — the token being predicted: the answer.
- **pair** — one (context, target). One prediction. Also called an
  **example**.
- **`Pairs`** — all the pairs of a corpus, held as two arrays: `contexts` and
  `targets`; reached through `make_pairs`, which returns one. Not exported by the
  package façade, so a caller that needs the type by name imports it from
  `kernel/lm/pairs.py`.
- **stream length** — the number of ids in the stream. Equal to the number of
  predictions, and therefore to the number of pairs. Written `len(stream)` and
  called **`N`**.
- **pairs per unit** — how many pairs one unit contributes: a unit of `m` tokens
  contributes `m + 1` pairs, because every token is predicted once and EOS is predicted too.
- **split** — the division of pairs into training and validation sets. Made per
  unit, so every pair of a unit lands on the same side. Which units are held out
  is a stride over the corpus - every n-th unit, offset by the seed - so the
  split follows corpus order: reordering the corpus can move a unit across, and
  appending can only add to the held-out set. How it works is documented on
  :meth:`Pairs.split` and `kernel/_holdout.py`.
- **batch** — a stack of pairs handed to the model at once.

## Shapes

As a compact representation, the `contexts` are packaged as a matrix with one row per
prediction and one column per slot.

```
                   col 0   col 1   col 2        w = 3 columns = slots,
row 1 (target a)    BOS     BOS     BOS                 oldest ... newest
row 2 (target b)    BOS     BOS      a
row 3 (target EOS)  BOS      a       b
     ^ N = 3 rows = predictions
```

- **row** — one prediction: one pair, i.e. its context.
- **column** — one slot: the same distance back from the target, in every row.

`Pairs` keeps the targets **outside** this array: `contexts` is `(N, w)` and
`targets` is `(N,)`. The target of row *i* is the last column of row *i+1*,
wherever both rows belong to the same unit - that is what the slide does. The
pair builder does not exploit it: it writes both arrays from the same
`[BOS, s1..sm, EOS]` sequence, so a row's target stays unambiguous even when its
context is padded out with BOS, or when a unit is shorter than `w`.

## Model

- **architecture** — a buildable recipe: a Keras graph plus its own
  hyperparameters (`Bigram`, `MLP`, `LSTM`). No weights, no tokenizer.
- **model** — an instance of an architecture with a tokenizer and weights: what
  `fit` returns, and what generates, evaluates and saves. The architecture is
  the recipe; the model is one fitted result of it.
- **hyperparameter** — a setting for an architecture, e.g. `context_width`,
  `embedding_dim`, or the LSTM's `units`. Enumerable through `get_params()`.
- **parameter** — a number the model learns from the data (embeddings, weights,
  biases), as against a **hyperparameter**, which is chosen. `parameter_count()`
  counts these.
- **embedding** — the learned vector for one token id; `embedding_dim` is how
  long it is.
- **order** — how many tokens a prediction involves: its `w` ids of history plus
  the target, so `order = w + 1`. A bigram conditions on one token, so its order
  is fixed at 2; an `MLP` or `LSTM` order follows its context width.
- **n-gram** — a model of order `n`, conditioning on `n - 1` ids: `Bigram` is
  the `n = 2` member. `MLP` and `LSTM` are trainable relatives of the same
  family, whose order is chosen instead of fixed.
- **logits** — one score per token in the vocabulary, unnormalised.
- **softmax** — turns logits into probabilities: positive and summing to one. It
  is applied in the loss and in the sampler, never in the Keras graph.
- **loss** — the number training minimises: the cross-entropy between the
  model's distribution over the vocabulary and the actual target, averaged over
  all predictions in the set being measured (train or validation).
- **optimizer** — the rule that turns the loss into weight updates. Adam here,
  with the learning rate as its step size.
- **perplexity** — `e^loss`: roughly, how many tokens the model is choosing
  between at each step. 1 is perfect and the vocabulary size is a uniform guess.
  Reported for the training and the validation pairs separately.
- **in-domain** / **out-of-domain** — text drawn from the corpus the model was
  trained on, versus text from anywhere else. The two perplexities are the
  comparison.
- **learning rate** — how far one optimizer step moves the weights.
- **batch size** — how many pairs one optimizer step averages over; the `batch`
  is that stack of pairs.
- **validation fraction** — the share of *units* held out for the split, not of
  pairs.
- **seed** — the number a run's random choices are derived from: weight
  initialisation and the split, so the same seed and inputs repeat a run.
  `generate` takes a seed of its own for sampling.
- **epoch** — one pass over all the training pairs. The per-epoch stats carry the
  loss, the validation loss and both perplexities, plus a sample if the caller's
  hook adds one.
- **prompt** — the user's text, tokenized and trimmed to its last unit, used as
  the first context of a generation.
- **generation** — repeat: context → logits → pick one id → append it → drop the
  oldest id. It stops at EOS or at the token cap, and a sampled BOS is dropped
  rather than appended.
- **temperature** — how the picking is done: `0` takes the best id, and higher
  values flatten the distribution.

## Relations

```
w = 1                     Bigram: a constant, not a parameter
w = a hyperparameter      MLP, LSTM
order = w + 1             Bigram 2; the others chosen with their width
N = sum(pairs per unit)   independent of w
contexts.shape = (N, w)
```

## Example: one small corpus, three architectures

Corpus: two units, `a b` and `c`. The `MLP` and `LSTM` are given a
context width of 3; the `Bigram` has no width to be given and reads 1.

```
stream   = [a, b, EOS, c, EOS]        len(stream) = 5 = N = 5 pairs
unit "a b": seq = [BOS, a, b, EOS]    pairs: 3  (2 tokens + EOS)
unit "c":   seq = [BOS, c, EOS]       pairs: 2  (1 token + EOS)
```

| # | target | `Bigram` context (`w = 1`) | `MLP` context (`w = 3`) | `LSTM` context (`w = 3`) |
|---|---|---|---|---|
| 1 | `a` | `[BOS]` | `[BOS, BOS, BOS]` | `[BOS, BOS, BOS]` |
| 2 | `b` | `[a]` | `[BOS, BOS, a]` | `[BOS, BOS, a]` |
| 3 | `EOS` | `[b]` | `[BOS, a, b]` | `[BOS, a, b]` |
| 4 | `c` | `[BOS]` | `[BOS, BOS, BOS]` | `[BOS, BOS, BOS]` |
| 5 | `EOS` | `[c]` | `[BOS, BOS, c]` | `[BOS, BOS, c]` |

What the table shows:

- **Width belongs to the model.** The `MLP` and `LSTM` were set to 3, the
  `Bigram` cannot be set at all; it reads 1 slot because that is what a bigram
  is.
- **Order follows `w`.** These five predictions are order 2 for `Bigram` and
  order 4 for the other two: the same targets, asked of different models.
- **Width changes columns only.** The five targets and the split are identical
  in all three columns, so what differs is only how much history each model
  reads. The bigram is fed exactly the pair a count table would use.
- **All BOS means "nothing came before".** Rows 1 and 4 are all BOS because BOS
  opening a unit's context and BOS padding a short context out are the *same
  id*. Those rows are trained, not discarded, so a unit's opening token is
  learnable.

And why the architectures differ at all, with `V` = `vocab_size`, `d` =
`embedding_dim`, `h` = `hidden_units`:

| architecture | reads | order | parameters |
|---|---|---|---|
| `Bigram` | 1 slot | 2 | `V²`, independent of `w` |
| `MLP` | all `w` slots | `w + 1` | grows with `w` (first dense layer takes `w*d` inputs) |
| `LSTM` | all `w` slots | `w + 1` | independent of `w` (shared weights over steps) |
