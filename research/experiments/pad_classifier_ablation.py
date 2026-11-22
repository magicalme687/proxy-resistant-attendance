"""
PAD classifier ablation on the cached 16-D CelebA-Spoof features.

Compares classifiers trained on the *same* 16-D feature vectors under the same
5-fold stratified CV, so the choice of calibrated HGB is justified by data.
Outputs results/pad_classifier_ablation.csv
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2


def models(seed):
    hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=31,
                                         min_samples_leaf=20, random_state=seed)
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RBF-SVM (Platt)": make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True, random_state=seed)),
        "Random forest (300)": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "HGB (uncalibrated)": hgb,
        "HGB + Platt (proposed)": CalibratedClassifierCV(estimator=hgb, method="sigmoid", cv=3),
    }


def main():
    d = np.load(os.path.join(RES, "celeba_features_cache.npz"))
    X, y = d["X"], d["y"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name in models(0):
        oof = np.zeros(len(y))
        times = []
        for k, (tr, va) in enumerate(skf.split(X, y)):
            m = models(42 + k)[name]
            m.fit(X[tr], y[tr])
            t0 = time.perf_counter()
            oof[va] = m.predict_proba(X[va])[:, 1]
            times.append((time.perf_counter() - t0) / len(va) * 1000)
        e = eer(y, oof)
        brier = float(np.mean((oof - y) ** 2))
        rows.append({"classifier": name, "roc_auc": round(roc_auc_score(y, oof), 4),
                     "eer_pct": round(100 * e, 2), "brier": round(brier, 4),
                     "infer_ms_per_sample": round(float(np.mean(times)), 4)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(RES, "pad_classifier_ablation.csv"), index=False)


if __name__ == "__main__":
    main()
