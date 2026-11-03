"""Ceiling probe (kept as the record of the measurement that motivated the
colour trap group `G`): features-only vs features+colour, z-scored, via the
kernel's own group G. Run in the Orange env (needs sklearn):

    /Applications/Orange.app/Contents/MacOS/python demo/probe_colour.py
"""
import sys, warnings; sys.path.insert(0, "tests")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn.neural_network")
import numpy as np, cv2
from pathlib import Path
from Orange.data import Domain, DiscreteVariable, ContinuousVariable, Table
from nn_kernel.features import extract_features, FeatureOptions, group_names
from Orange.evaluation import CrossValidation
from Orange.classification.neural_network import NNClassificationLearner
from sklearn.metrics import confusion_matrix

NAMES_36 = [n for g in "ABCDEF" for n in group_names(g)]
NAMES_40 = NAMES_36 + [n for g in "G" for n in group_names(g)]

root = Path("data/clouds")
classes = sorted(d.name for d in root.iterdir() if d.is_dir())
X36, X40, y = [], [], []
for ci, cls in enumerate(classes):
    for path in sorted((root / cls).glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        vec = extract_features(img, FeatureOptions(working_size=128), groups="ABCDEFG")
        X36.append([vec[n] for n in NAMES_36])
        X40.append([vec[n] for n in NAMES_40])
        y.append(ci)
X36, X40, y = np.asarray(X36), np.asarray(X40), np.asarray(y)

def make_table(Xv, names):
    dom = Domain([ContinuousVariable(n) for n in names],
                 [DiscreteVariable("genus", values=classes)])
    return Table.from_numpy(dom, Xv, y)

def run(label, Xv, names, hidden=(10,), iters=500):
    learner = NNClassificationLearner(hidden_layer_sizes=hidden, max_iter=iters)
    res = CrossValidation(make_table(Xv, names), [learner], k=5, store_models=False)
    pred = res.predicted[0] if res.predicted.ndim > 1 else res.predicted
    cm = np.asarray(confusion_matrix(res.actual, pred), dtype=int)
    print(f"== {label}: CA = {cm.diagonal().sum()/cm.sum():.3f}")
    print("   confusion:\n", cm)
    print("   predicted totals:", cm.sum(axis=0))

def zscore(Xv):
    return (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-12)

run("36 features, raw", X36, NAMES_36)
run("36 features, z-scored", zscore(X36), NAMES_36, hidden=(20,), iters=1000)
run("40 (with colour G), z-scored", zscore(X40), NAMES_40, hidden=(20,), iters=1000)