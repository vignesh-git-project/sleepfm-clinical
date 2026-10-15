"""
Fit a staging calibration from PSG-labelled recordings and write staging_calibration.json.

Expected input folder (one pair of files per subject):
    <data_dir>/<subject>_logits.npy   raw epoch logits saved by inference.py (n_epochs x 5)
    <data_dir>/<subject>_labels.npy   PSG stages as ints per 30 s epoch:
                                      0=Wake, 1=N1, 2=N2, 3=N3 (merge N4 into N3), 4=REM

Usage:
    python fit_staging_calibration.py <data_dir> [--out staging_calibration.json] [--folds 7]

Reports leave-subject-out agreement (before vs after) and then fits the final model
on all subjects. Check the leave-subject-out numbers before using the JSON.
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

STAGES = ['Wake', 'N1', 'N2', 'N3', 'REM']


def load(data_dir):
    X, y, g = [], [], []
    for lp in sorted(glob.glob(os.path.join(data_dir, '*_logits.npy'))):
        subj = os.path.basename(lp)[:-len('_logits.npy')]
        yp = os.path.join(data_dir, f'{subj}_labels.npy')
        if not os.path.exists(yp):
            print(f'skip {subj}: no labels file')
            continue
        logits, labels = np.load(lp), np.load(yp).astype(int)
        n = min(len(logits), len(labels))  # align epochs by truncating to the shorter one
        if len(logits) != len(labels):
            print(f'{subj}: logits {len(logits)} vs labels {len(labels)} epochs, truncating to {n}')
        X.append(logits[:n])
        y.append(labels[:n])
        g.append(np.full(n, len(g)))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def report(name, y, pred):
    print(f'\n== {name} ==')
    print('kappa:', round(cohen_kappa_score(y, pred), 3))
    print('macro F1:', round(f1_score(y, pred, average='macro'), 3))
    print('per-stage F1:', dict(zip(STAGES, np.round(f1_score(y, pred, average=None, labels=range(5)), 3))))
    print('confusion (rows=PSG, cols=device):\n', confusion_matrix(y, pred, labels=range(5)))
    prop_psg = np.bincount(y, minlength=5) / len(y)
    prop_dev = np.bincount(pred, minlength=5) / len(pred)
    print('stage share PSG   :', dict(zip(STAGES, np.round(prop_psg * 100, 1))))
    print('stage share device:', dict(zip(STAGES, np.round(prop_dev * 100, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--out', default='staging_calibration.json')
    ap.add_argument('--folds', type=int, default=7)
    a = ap.parse_args()

    X, y, g = load(a.data_dir)
    n_subj = len(np.unique(g))
    print(f'{n_subj} subjects, {len(y)} epochs')
    assert set(np.unique(y)) == set(range(5)), 'labels must contain all 5 classes 0-4'

    # Baselines
    report('Raw logits (argmax)', y, X.argmax(1))
    centered = np.vstack([X[g == s] - X[g == s].mean(0) for s in np.unique(g)])
    report('Previous behaviour: per-night centered logits', y, centered.argmax(1))

    # Leave-subject-out calibration
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=min(a.folds, n_subj)).split(X, y, g):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    report('Leave-subject-out recalibration', y, pred)

    # Final fit on all subjects
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    assert list(clf.classes_) == list(range(5))
    with open(a.out, 'w') as f:
        json.dump({'coef': clf.coef_.tolist(), 'intercept': clf.intercept_.tolist(),
                   'stages': STAGES, 'n_subjects': int(n_subj)}, f)
    print(f'\nWrote {a.out}')


if __name__ == '__main__':
    main()
