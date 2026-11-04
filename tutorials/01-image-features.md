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
2. Add **Image Feature Extractor**, set the working size to 56 (type the
   number and press Enter), and connect *Import Images* **Data** to its
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

The first extraction decodes every image and computes all feature groups once;
a progress bar reports both phases. After that, switching groups reuses the
cached raw features, so the ablation is quick. Changing preprocessing reuses
the decoded images but recomputes their features without reading the files
again.

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

Hover any group checkbox to see what it measures. In **Image Feature
Extractor**, switch a whole group off. The feature table is recomputed from the
cache.

Try this sequence:

1. Baseline: all groups A-F on.
2. Turn **E: topology** off. The hole count and stroke crossings disappear.
3. Turn **E** back on and turn **D: texture** off.
4. Turn **B: profile summaries** off as well.

Watch **Test & Score** and the confusion matrix after each change. For digits,
topology and spatial mass usually matter most; texture helps less than it does
for clouds. Record the group combination that works best and the one that
explains the most errors.

## 5. Browse the images

The main area of **Image Feature Extractor** shows every input image as a
thumbnail. The header reports the total count, and clicking a thumbnail selects
it. Use the grid to spot blank, corrupted or unexpected images before training.
Selecting an image does not change the extracted features.

The **Feature values** panel on the right lists the selected image's features,
grouped A-G. Groups that are switched off stay visible but greyed out. Click a
feature to read its definition and see a histogram across the data, split by
class, with a red line at the selected image's value. This is the quickest way
to see whether a feature separates the labels before trusting it in a model.

If you prefer a standard table, connect **Features** to a **Data Table**'s
*Data* input and **Selected image** to its *Data Subset* input. The table then
opens with the selected image highlighted.

## 6. Optional: scale features correctly

Some learners are sensitive to feature scale. Do not z-score the whole output
table before evaluation; that leaks statistics across validation folds.

Instead:

1. Add **Preprocess** and put a **Normalize** step in its pipeline.
2. Connect *Image Feature Extractor* **Features** → *Preprocess* **Data**.
3. Connect *Preprocess* **Preprocessor** → *Test & Score* **Preprocessor**.

Test & Score then fits the normalisation inside each fold. The same
**Preprocessed Data** output can feed a learner outside Test & Score when you
do not need cross-validation.

## 7. Optional: raw-pixel baseline

Add **Pixel Features** and connect *Import Images* **Data** → *Pixel Features*
**Images**, and *Pixel Features* **Features** → *Test & Score* **Data**. Leave
the dimensions at 10. This gives 100 raw luminance columns per image and no
spatial encoding; it is the deliberately plain baseline against which the
handcrafted features can be compared.

Raise the dimensions to 14 or 28 and compare the accuracy and the number of
columns. The raw-pixel model has no idea that adjacent pixels are neighbours,
only that each pixel has its own input weight.

## Checkpoint

- The feature output has 36 columns plus the inherited class and image path.
- Turning groups off changes both the table and the evaluation.
- You can point to a confused digit pair and name the feature that explains it.

## What next

- Clouds, colour and baseline learners: [02 - Colour shortcut](02-colour-shortcut.md)
- The language-model stream: [03 - Tokenizers](03-tokenizers.md)
