# 02 - Colour shortcut and baselines

Build the cloud-genus workflow, compare feature groups, and see why colour is
the deliberate shortcut.

## Prepare the corpus

```bash
uv run python demo/fetch_clouds.py
```

This writes `data/clouds/<genus>/` and a credits file. The images are for
personal, non-commercial teaching use and are not part of the repository's
code licence.

## 1. Open or build the workflow

**Quick start:** open `demo/cloud_features.ows` with **File > Open**. The
Image Feature Extractor is set to working size 128 and groups A-F.

If the Import Images path is stale, open **Import Images**, click the folder
button and choose `data/clouds`.

**From scratch:**

1. **Import Images** → folder `data/clouds`.
2. **Image Feature Extractor**, working size 128.
3. **Test & Score**, **Confusion Matrix**, **Data Table**, **Column
   Statistics**.
4. **Neural Network**, **Tree** and **Logistic Regression**.

Connect:

- *Import Images* **Data** → *Image Feature Extractor* **Images**
- *Image Feature Extractor* **Features** → *Test & Score* **Data**
- *Neural Network* **Learner** → *Test & Score* **Learner**
- *Tree* **Learner** → *Test & Score* **Learner**
- *Logistic Regression* **Learner** → *Test & Score* **Learner**
- *Test & Score* **Evaluation Results** → *Confusion Matrix*
  **Evaluation Results**
- *Image Feature Extractor* **Features** → *Data Table* **Data**
- *Image Feature Extractor* **Features** → *Column Statistics* **Data**

## 2. Baseline: luminance features only

Leave group **G: colour** unticked. Run **Test & Score** with cross-validation
(5 folds).

Read the confusion matrix by genus. Which genera are confused? Close-up
photographs and mixed-sky scenes often produce the same profile and texture
signatures. This is the honest difficulty of hand-crafted features.

## 3. Switch on the shortcut

Tick **G: colour - the shortcut trap**. The output grows from 36 to 40 columns.

Look at the new features in **Data Table**:

- `bluish_fraction` - fraction of pixels bluer than red;
- `blueness_mean` - overall blue-red balance;
- `saturation_mean` - colourfulness;
- `greenery_fraction` - ground clutter.

Run **Test & Score** again. Accuracy usually rises. The model has found a
shortcut: "is the image blue?" is a strong proxy for "is this a sky cloud
photo?", without learning any spatial structure. The group is off by default
for exactly this reason.

## 4. Compare learners on identical features

With group G on, look at the three rows in **Test & Score**:

- Neural Network;
- Tree;
- Logistic Regression.

If all three land within a small range, the *features* carried the
intelligence and the neural network is mostly ceremony. If the network clearly
pulls ahead, inspect the confusion matrix and ask which nonlinear combination
it might have found. Either result is a finding.

## 5. Ablation

Try these combinations and note what happens:

1. A-F only (no colour).
2. A-F plus G.
3. G only.
4. D + G.
5. B + D + G.

Use the group checkboxes; the feature table and evaluation update
automatically. Group-accuracy notes are more useful than a single score: they
show which physical properties of the image are doing the work.

## Checkpoint

- You can explain why colour is called a trap.
- You can show a case where several learners agree, and say what that means.
- You can point at a feature group and predict whether removing it will hurt.

## What next

- Inspect the tokenizers behind the language models:
  [03 - Tokenizers](03-tokenizers.md)
- Try the same ablation idea on digits:
  [01 - Image features](01-image-features.md)
