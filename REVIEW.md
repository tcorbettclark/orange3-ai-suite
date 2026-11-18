# Reviewing a change

A checklist for structural changes: new modules, moved code, renamed things, split
or merged classes. Use it twice — before starting, so the shape is chosen rather
than discovered, and before merging, so the shape is checked rather than assumed.

Every question is meant to be answered with a fact about the code. "I think so" is
not an answer; if you cannot answer one, that is the finding.

## Words

- Does any word carry more than one meaning? List the nouns the code uses and give
  each exactly one meaning per layer. Where two layers need different meanings, they
  need different words.
- Does any name describe the material rather than the product? Names like *data*,
  *values*, *pixels*, *grid*, *utils*, *lib* distinguish nothing in a directory of
  similar files. Say what comes out in one sentence: if the name does not predict
  that sentence, it is the wrong name.
- Does a name have to cover more than one verb? A name that needs "and" is a missing
  split, not a naming problem.
- Is each technical term the one the formula implements? Prose and arithmetic must
  use the same word, including units and qualifiers.
- Does the same concept appear under two names? Synonyms are as costly as
  homonyms, and harder to notice.

## Steps and boundaries

- Is the mechanism in the library and the purpose in the caller? A caller composes
  steps; the library supplies them. A caller that reimplements a step is missing a
  function.
- Does any function hide a pipeline? If its steps have their own failure modes or
  their own tests, they are functions of their own.
- Is anything shared at the wrong level? If two packages use something from inside a
  third, it belongs a level higher, where both can reach it as a sibling.
- Does any file serve two layers at once? If you cannot name the single concept it
  serves, either it is two files, or the concept is "the arguments of one call" — and
  then the docstring should say exactly that.
- Is any policy spelled twice? Tolerances, interpolation, ordering, naming, rounding.
  One spelling is a decision; two is drift waiting to happen.
- Does any default decide something on the caller's behalf? A default is an
  invisible decision. Either require the argument or make the default explicit and
  documented.

## Layout and dependency

- Build the table first, then read it instead of the code: file, one-line purpose,
  imports, imported by, when it runs.
- Where does an arrow point "upward" — a sibling reaching into a subpackage, a leaf
  importing its parent, a cycle? Each is either a misplaced file or a missing module.
- Which files have exactly one caller? A helper of one caller belongs with that
  caller — unless it is a product with its own tests. Decide which, and say so.
- Are the private/public marks consistent? The mechanical test: does anything outside
  the package import it? Then public; otherwise private. Enforce it with a test rather
  than a convention.
- What axis organises each file — pipeline stage, deliverable, view, or reuse? A
  reader should be able to tell from the directory. A file organised by reuse needs
  to say so, or it reads as "miscellaneous".
- Does the tree match the rest of the project? A domain that is split by abstraction
  level in one place and by subject in another makes both harder to find.
- Is anything maintained twice — a page beside a catalogue, a list beside a
  directory? Generate it or test it, never both.

## Truth and evidence

- For every claim of the form "always" or "never", read the branch. Intent written as
  fact is a defect, and it makes every neighbouring sentence need checking.
- Does every branch exist in a test? And does the test pin the policy or only one
  outcome? A branch with no test is a branch nobody has decided about.
- Which suite runs this code? Pure computation belongs in the kernel, testable
  without Qt and without Orange; anything that needs a widget to be tested has been
  put in the wrong layer.
- If a recommendation changes behaviour, quantify it before recommending: what moves,
  by how much, and would any test notice? An unquantified claim is an opinion, and the
  measurement is usually shorter than the argument.
- For a pure move or rename, is there a diff proving nothing changed? For an
  intentional change, is the measured effect named in the commit message? Without
  one, a refactor and a regression look identical.
- Is any correspondence held only by convention — names beside values, a roster
  beside a directory, membership beside a catalogue? Make it structural, so one call
  produces both, or test it. Never neither.

## Finishing

- Can each decision be reverted alone? One commit per structural step. A rename
  riding on a move cannot be reviewed.
- What runs at import time? Validation is fine; work is not. The "when it runs"
  column of the table catches this, and catches accidental caching with it.
- Is the prose still true after the move — paths, module references, generated pages,
  tutorial text? Generated content cannot go stale; authored content needs a sweep.
- Would a newcomer reading only the file names and the docstrings arrive at the same
  mental model you have? If not, the code says one thing and the words another.
