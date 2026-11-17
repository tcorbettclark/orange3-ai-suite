# Glossary

## Processes

- **tokenizer**: `text` → `stream` — the whole corpus as one list of ids, EOS
  after every unit.
- **pair builder**: `stream` → `pairs` — one context and one target per prediction.
- **model**: `context` → `logits` — one score per token in the vocabulary.
- **sampler**: `logits` → `id` — one id per step of a generation.

## Text

- **text** — a string a person wrote.
- **corpus** — all the text being trained on.
- **unit** — one self-contained piece of text. Currently a unit is a sentence.

## Tokens

Tokens are built by the tokenizer (`kernel/lm/tokenizer.py`).

- **token** — the smallest piece the tokenizer emits: one character (char level)
  or one word or punctuation mark (word level).
- **id** — a token's integer. Opaque: only the tokenizer's table gives it
  meaning, and ids never mean anything across two tokenizers.
- **vocabulary** — the set of tokens the tokenizer knows. **`vocab_size`** — how
  many, which is also how many logits the model emits.
- **stream** — the corpus as one flat list of ids. Each unit ends with EOS. The
  stream is the only input to the pair builder.
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
- **frequency table** — the top `n` real tokens as (rank, token, count), which is
  what the vocabulary viewer shows.
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
  reads. A hyperparameter of `MLPWindow` and `LSTM`, enumerable through
  `get_params()`, and the constant 1 for `Bigram`, which has no such parameter: a
  model that reads more history is an n-gram of higher order, a different model
  rather than a setting.

## Pairs

Pairs are built by the pair builder (`kernel/lm/pairs.py`).

- **target** — the token being predicted: the answer.
- **pair** — one (context, target). One prediction. Also called an
  **example**.
- **`Pairs`** — all the pairs of a corpus, held as two arrays: `contexts` and
  `targets`.
- **stream length** — the number of ids in the stream. Equal to the number of
  predictions, and therefore to the number of pairs. Written `len(stream)` and
  called **`N`**.
- **pairs per unit** — how many pairs one unit contributes: a unit of `m` tokens
  contributes `m + 1` pairs, because every token is predicted once and EOS is predicted too.
- **split** — the division of pairs into training and validation sets. Made per
  unit, so the pairs from a unit are either in the training or the validation
  set, not both. Which units go where is decided by a salted hash of each unit's
  own ids, so the split is deterministic under a seed, independent of the order
  the corpus arrived in, and stable when more text is appended.
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
`targets` is `(N,)`. The id in the last column of row *i* is row *i+1*'s target
wherever both rows belong to the same unit - that is what the slide does. The
pair builder does not exploit it: it writes both arrays from the same
`[BOS, s1..sm, EOS]` sequence, so a row's target stays unambiguous even when its
context is padded out with BOS, or when a unit is shorter than `w`.

## Model

- **architecture** — one buildable model class: `Bigram`, `MLPWindow`, `LSTM`.
- **hyperparameter** — a setting for an architecture, e.g. `context_width`,
  `units` or `hidden_units`. Enumerable through `get_params()`.
- **order** — how many tokens a prediction involves: its `w` ids of history plus
  the target, so `order = w + 1`. A bigram conditions on one token, so its order
  is fixed at 2; an `MLPWindow` or `LSTM` order follows its context width.
- **n-gram** — a model of order `n`, conditioning on `n - 1` ids: `Bigram` is
  the `n = 2` member. `MLPWindow` and `LSTM` are trainable relatives of the same
  family, whose order is chosen instead of fixed.
- **logits** — one score per token in the vocabulary, unnormalised. Softmax is
  computed where it is needed: inside the training loss and in the sampler.
- **loss** — the number training minimises: the cross-entropy between the
  model's distribution over the vocabulary and the actual target, averaged over
  all predictions in the set being measured (train or validation).
- **perplexity** — `e^loss`: roughly, how many tokens the model is choosing
  between at each step. 1 is perfect and the vocabulary size is a uniform guess.
  Reported for the training and the validation pairs separately.
- **epoch** — one pass over all the training pairs. The checkpoint table has one
  row per epoch, with the loss, the validation loss, both perplexities and an
  optional sample.
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
w = a hyperparameter      MLPWindow, LSTM
order = w + 1             Bigram 2; the others chosen with their width
N = sum(pairs per unit)   independent of w
contexts.shape = (N, w)
```

## Example: one small corpus, three architectures

Corpus: two units, `a b` and `c`. The `MLPWindow` and `LSTM` are given a
context width of 3; the `Bigram` has no width to be given and reads 1.

```
stream   = [a, b, EOS, c, EOS]        len(stream) = 5 = N = 5 pairs
unit "a b": seq = [BOS, a, b, EOS]    pairs: 3  (2 tokens + EOS)
unit "c":   seq = [BOS, c, EOS]       pairs: 2  (1 token + EOS)
```

| # | target | `Bigram` context (`w = 1`) | `MLPWindow` context (`w = 3`) | `LSTM` context (`w = 3`) |
|---|---|---|---|---|
| 1 | `a` | `[BOS]` | `[BOS, BOS, BOS]` | `[BOS, BOS, BOS]` |
| 2 | `b` | `[a]` | `[BOS, BOS, a]` | `[BOS, BOS, a]` |
| 3 | `EOS` | `[b]` | `[BOS, a, b]` | `[BOS, a, b]` |
| 4 | `c` | `[BOS]` | `[BOS, BOS, BOS]` | `[BOS, BOS, BOS]` |
| 5 | `EOS` | `[c]` | `[BOS, BOS, c]` | `[BOS, BOS, c]` |

What the table shows:

- **Width belongs to the model.** The `MLPWindow` and `LSTM` were set to 3, the
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
| `MLPWindow` | all `w` slots | `w + 1` | grows with `w` (first dense layer is `w*d` wide) |
| `LSTM` | all `w` slots | `w + 1` | independent of `w` (shared weights over steps) |
