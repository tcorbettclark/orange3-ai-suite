"""Headless diagnostic: reproduce the canvas cloud confusion matrix and test
the feature-scaling hypothesis (spec §3 engineering note: 'scale features
inside the widget — the MLP is scale-sensitive')."""
import sys, warnings; sys.path.insert(0, "tests")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.neural_network")
import numpy as np, cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
from Orange.data import Domain, DiscreteVariable, ContinuousVariable, Table
from nn_kernel.features import extract_features, FeatureOptions, group_names

root = Path("data/clouds")
classes = sorted(d.name for d in root.iterdir() if d.is_dir())
X, y = [], []
for ci, cls in enumerate(classes):
    for path in sorted((root / cls).glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        vec = extract_features(img, FeatureOptions(working_size=128), groups='ABCDEFG')
        X.append([vec[n] for n in [n for g in 'ABCDEFG' for n in group_names(g)]])
        y.append(ci)
X = np.asarray(X); y = np.asarray(y)

def make_table(Xv):
    dom = Domain([ContinuousVariable(n) for n in [n for g in 'ABCDEFG' for n in group_names(g)]],
                 [DiscreteVariable("genus", values=classes)])
    return Table.from_numpy(dom, Xv, y)

from Orange.classification.neural_network import NNClassificationLearner
from Orange.evaluation import CrossValidation
from Orange.classification.tree import SklTreeLearner

def run(label, Xv, learner):
    res = CrossValidation(make_table(Xv), [learner], k=5, store_models=False)
    pred = res.predicted if res.predicted.ndim == 1 else res.predicted[0]
    cm = np.asarray(confusion_matrix(res.actual, pred), dtype=int)
    ca = float(np.diag(cm).sum()) / cm.sum()
    print(f"== {label}: CA = {ca:.3f}  (rows=true, cols=pred; {classes})")
    print(cm)
    print("predicted totals per class:", cm.sum(axis=0))

run("RAW features (canvas today)", X,
    NNClassificationLearner(hidden_layer_sizes=(10,), max_iter=500))
Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
run("Z-SCORED features", Xs,
    NNClassificationLearner(hidden_layer_sizes=(10,), max_iter=500))
run("Tree on raw (scale-insensitive)", X, SklTreeLearner())
print("majority baseline:", round(max(np.bincount(y, minlength=len(classes))) / len(y), 3))