# 02 - Colour shortcut and baselines

Build the cloud-genus workflow, compare feature groups, and see why colour is
the deliberate shortcut.

## Prepare the corpus

```bash
orange-ai-assets fetch ccsn-clouds
```

This prepares the five-genus CCSN subset (Ac, Sc, Cu, Cb, Ci) and its
`CREDITS.md`. The CCSN database is CC0; cite Zhang et al. (2018), *CloudNet*.
The command prints the prepared root.

## 1. Open or build the workflow

**Quick start:** open `demo/cloud_features.ows` with **File > Open**. The
Image Features is set to working size 128 and groups A-F.

If the Import Images path is stale, open **Import Images**, click the folder
button and choose the prepared CCSN root.

**From scratch:**

1. **Import Images** → the prepared CCSN root.
2. **Image Features**: set working size to 128 (type the number and
   press Enter).
3. **Test & Score**, **Confusion Matrix**, **Data Table**, **Column
   Statistics**.
4. **Neural Network**, **Tree** and **Logistic Regression**.

Connect:

- *Import Images* **Data** → *Image Features* **Images**
- *Image Features* **Features** → *Test & Score* **Data**
- *Neural Network* **Learner** → *Test & Score* **Learner**
- *Tree* **Learner** → *Test & Score* **Learner**
- *Logistic Regression* **Learner** → *Test & Score* **Learner**
- *Test & Score* **Evaluation Results** → *Confusion Matrix*
  **Evaluation Results**
- *Image Features* **Features** → *Data Table* **Data**
- *Image Features* **Features** → *Column Statistics* **Data**

## 2. Baseline: luminance features only

Leave group **G: colour** unticked. Run **Test & Score** with cross-validation
(5 folds).

Read the confusion matrix by genus. Which genera are confused? Close-up
photographs and mixed-sky scenes often produce the same profile and texture
signatures. This is the honest difficulty of hand-crafted features.

## 3. Switch on the shortcut

Tick **G: colour**. The output grows from 34 to 38 columns.

Look at the new features in **Data Table**:

- `bluish_fraction` - fraction of pixels bluer than red;
- `mean_blueness` - overall blue-red balance;
- `mean_saturation` - colourfulness;
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

- You can explain why colour is treated as a shortcut.
- You can show a case where several learners agree, and say what that means.
- You can point at a feature group and predict whether removing it will hurt.

## What next

- Inspect the tokenizers behind the language models:
  [03 - Tokenizers](03-tokenizers.md)
- Try the same ablation idea on digits:
  [01 - Image features](01-image-features.md)
