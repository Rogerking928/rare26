"""在凍結特徵上訓練線性分類器，用官方 metric 評估。

驗證設計：官方驗證／測試集來自 BONSAI 的 12 家中心，訓練集只有 2 家。
所以真正要量的是「跨中心是否還撐得住」，不是同分布的隨機切分。
兩者都跑：
  - LOCO（leave-one-center-out）：跨中心，悲觀但貼近實測
  - CV5：分層 5 折，樂觀，只拿來看訓練訊號在不在
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from scorer import official_score, fpr_at_sensitivity  # noqa: E402

warnings.filterwarnings("ignore")


def load(tag):
    d = np.load(ROOT / "runs" / f"feats_{tag}.npz", allow_pickle=True)
    return d["X"], d["y"].astype(int), d["center"].astype(int)


def fit_predict(Xtr, ytr, Xte, C, balanced):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(
        C=C, max_iter=5000,
        class_weight="balanced" if balanced else None,
    ).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def evaluate(y, s, seed=0):
    r = official_score(y, s, n_iterations=400, rng=np.random.default_rng(seed))
    fpr, _ = fpr_at_sensitivity(y, s, 0.9)
    return r["Score"], r["AUROC Full Dataset"], fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet50_tta")
    ap.add_argument("--C", type=float, nargs="+",
                    default=[0.0001, 0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()

    X, y, center = load(args.tag)
    print(f"{args.tag}: X={X.shape}  正樣本={y.sum()}  "
          f"center_1={(center==1).sum()}  center_2={(center==2).sum()}\n")

    hdr = f"{'C':>8} | {'LOCO Score':>10} {'AUROC':>7} {'FPR@90':>8} | " \
          f"{'CV5 Score':>10} {'AUROC':>7} {'FPR@90':>8}"
    print(hdr); print("-" * len(hdr))

    for C in args.C:
        # --- LOCO：每個中心各當一次測試集，合併預測後一次評分
        s_loco = np.empty(len(y), float)
        for c in (1, 2):
            te = center == c
            s_loco[te] = fit_predict(X[~te], y[~te], X[te], C, args.balanced)
        # 兩個中心的 decision_function 尺度不同，各自轉成中心內的排名再合併，
        # 否則合併評分會被尺度差異汙染（metric 只看排序）
        s_loco_r = np.empty(len(y), float)
        for c in (1, 2):
            m = center == c
            s_loco_r[m] = np.argsort(np.argsort(s_loco[m])) / max(1, m.sum() - 1)
        a = evaluate(y, s_loco_r)

        # --- CV5：分層 5 折
        s_cv = np.empty(len(y), float)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            s_cv[te] = fit_predict(X[tr], y[tr], X[te], C, args.balanced)
        b = evaluate(y, s_cv)

        print(f"{C:>8g} | {a[0]:>10.4f} {a[1]:>7.4f} {a[2]:>8.4f} | "
              f"{b[0]:>10.4f} {b[1]:>7.4f} {b[2]:>8.4f}")


if __name__ == "__main__":
    main()
