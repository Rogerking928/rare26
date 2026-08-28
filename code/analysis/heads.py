"""同一組凍結特徵下，比較不同分類頭。

metric 只看排序，所以任何單調變換都無效（見 RESUME.md 結論 1）。
真正會改變排序的只有三件事：特徵怎麼標準化、正則化強度、以及多模型平均。
這裡就掃這三件事。
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from scorer import official_score, fpr_at_sensitivity  # noqa: E402
warnings.filterwarnings("ignore")


def rank01(v):
    v = np.asarray(v, float)
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def make(kind, C, balanced):
    cw = "balanced" if balanced else None
    if kind == "logreg":
        return LogisticRegression(C=C, max_iter=5000, class_weight=cw)
    if kind == "svm":
        return LinearSVC(C=C, max_iter=20000, class_weight=cw)
    if kind == "ridge":
        return RidgeClassifier(alpha=1.0 / C, class_weight=cw)
    raise ValueError(kind)


def fit_predict(Xtr, ytr, Xte, kind, C, balanced, l2norm, bag):
    """bag=k>1 時，用 k 折各訓一個模型再對排名平均（去年冠軍那類做法的簡化版）。"""
    if l2norm:
        Xtr, Xte = normalize(Xtr), normalize(Xte)
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)

    def one(idx):
        m = make(kind, C, balanced).fit(A[idx], ytr[idx])
        return m.decision_function(B)

    if bag <= 1:
        return one(np.arange(len(ytr)))
    outs = [rank01(one(tr)) for tr, _ in
            StratifiedKFold(bag, shuffle=True, random_state=0).split(A, ytr)]
    return np.mean(outs, axis=0)


def loco(X, y, center, **kw):
    out = np.empty(len(y), float)
    for c in np.unique(center):
        te = center == c
        out[te] = rank01(fit_predict(X[~te], y[~te], X[te], **kw))
    return out


def ev(y, s):
    r = official_score(y, s, n_iterations=400, rng=np.random.default_rng(0))
    return r["Score"], r["AUROC Full Dataset"], fpr_at_sensitivity(y, s, 0.9)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    d = np.load(ROOT / "runs" / f"feats_{args.tag}.npz", allow_pickle=True)
    X, y, center = d["X"], d["y"].astype(int), d["center"].astype(int)
    print(f"{args.tag}: X={X.shape}\n")

    hdr = f"{'頭':<8} {'C':>7} {'bal':>4} {'L2':>3} {'bag':>4} | {'Score':>7} {'AUROC':>7} {'FPR@90':>7}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for kind in ("logreg", "svm", "ridge"):
        for C in (0.0003, 0.001, 0.003, 0.01):
            for bal in (True, False):
                for l2 in (False, True):
                    for bag in (1, 5):
                        s = loco(X, y, center, kind=kind, C=C, balanced=bal,
                                 l2norm=l2, bag=bag)
                        sc, au, fp = ev(y, s)
                        rows.append((sc, kind, C, bal, l2, bag, au, fp))
    for sc, kind, C, bal, l2, bag, au, fp in sorted(rows, reverse=True)[:15]:
        print(f"{kind:<8} {C:>7g} {str(bal)[0]:>4} {str(l2)[0]:>3} {bag:>4} | "
              f"{sc:>7.4f} {au:>7.4f} {fp:>7.4f}")


if __name__ == "__main__":
    main()
