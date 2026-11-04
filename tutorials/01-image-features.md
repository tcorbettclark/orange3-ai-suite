# 01 - Image features

Build an interpretable image-classification workflow on MNIST, inspect the
features, and use group toggles as an ablation experiment.

## What you will build

```
Import Images -> Image Feature Extractor -> Test & Score -> Confusion Matrix
                                            -> Data Table
                                            -> Column Statistics
Neural Network -> Test & Score
```

## 1. Open or build the workflow

**Quick start:** open `demo/mnist_features.ows` with **File > Open**. The
Image Feature Extractor is already set to working size 56 and groups A-F.

If the Import Images path is stale, open **Import Images**, click the folder
button and choose `demo/mnist/train`.

**From scratch:**

1. Add **Import Images** and point it at `demo/mnist/train`.
2. Add **Image Feature Extractor** and connect *Import Images* **Data** to its
   **Images** input.
3. Add **Test & Score**, **Confusion Matrix**, **Data Table** and
   **Column Statistics**.
4. Add **Neural Network**.
5. Connect:
   - *Image Feature Extractor* **Features** → *Test & Score* **Data**
   - *Neural Network* **Learner** → *Test & Score* **Learner**
   - *Test & Score* **Evaluation Results** → *Confusion Matrix*
     **Evaluation Results**
   - *Image Feature Extractor* **Features** → *Data Table* **Data**
   - *Image Feature Extractor* **Features** → *Column Statistics* **Data**

## 2. Inspect the feature table

Open **Data Table**. You should see:

- one row per image;
- a class column (`0`-`9`) inherited from the folder names;
- a `image` meta column with the source path;
- 36 numeric feature columns from groups A-F.

Sort by `hole_count`. Digits with enclosed regions (`0`, `6`, `8`, `9`) score
higher than `1`, `2`, `3`, `5`, `7`. Look at
`foreground_area_fraction` next: it separates thin strokes from dense ones.

Open **Column Statistics** and click a few features. This is where you can see
whether a feature has a plausible range and whether a class is separable.

## 3. Train and evaluate

Open **Test & Score**:

1. Choose **Cross validation**.
2. Set the number of folds to 5.
3. Check that *Neural Network* appears in the learner list.
4. Read the classification accuracy (CA) and open **Confusion Matrix**.

Which digits are confused? The answer is usually pairs with similar topology
(`3`/`5`/`8`) or similar mass distribution (`4`/`9`). The point is not to beat
a convolutional network; it is to explain every mistake in terms of features.

## 4. Ablation: which groups carry the signal?

In **Image Feature Extractor**, switch a whole group off. The feature table is
recomputed automatically.

Try this sequence:

1. Baseline: all groups A-F on.
2. Turn **E: topology** off. The hole count and stroke crossings disappear.
3. Turn **E** back on and turn **D: texture** off.
4. Turn **B: profile summaries** off as well.

Watch **Test & Score** and the confusion matrix after each change. For digits,
topology and spatial mass usually matter most; texture helps less than it does
for clouds. Record the group combination that works best and the one that
explains the most errors.

## 5. Check the preview

In the main area of **Image Feature Extractor**, use **Image #** to select a
different image. The preview shows the preprocessed luminance and its row and
column profiles. This makes a feature like `row_profile_peaks` visible: the
peaks in the red or blue curve are the bands the feature counts.

## 6. Optional: feature scaling

Tick **Scale features** and compare. The option z-scores the whole output
table. That is convenient for learners that are sensitive to scale, but the
statistics come from the whole table, so it can leak information across
cross-validation folds. Keep that caveat in mind when reading the result.

## Checkpoint

- The feature output has 36 columns plus the inherited class and image path.
- Turning groups off changes both the table and the evaluation.
- You can point to a confused digit pair and name the feature that explains it.

## What next

- Clouds, colour and baseline learners: [02 - Colour shortcut](02-colour-shortcut.md)
- The language-model stream: [03 - Tokenizers](03-tokenizers.md)
