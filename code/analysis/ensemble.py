"""跨骨幹集成：各自訓一個線性頭，對 LOCO 預測的**排名**平均。

為什麼平均排名而不是平均機率：官方 metric 只看排序，
而不同骨幹的 decision_function 尺度天差地遠，直接平均會被尺度大的那個吃掉。
（RESUME.md 結論 1：單調變換對單一模型無效，但改變融合後的排序。）
"""
import itertools, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from scorer import official_score, fpr_at_sensitivity, paired_compare  # noqa: E402
warnings.filterwarnings("ignore")

TAGS = {
    "dinov2_224":  "vit_base_patch14_dinov2_224_tta",
    "convnextv2":  "convnextv2_tiny_tta",
    "resnet50":    "resnet50_tta",
}
C = 0.0003   # heads.py 掃出來的最佳設定：ridge + L2 + 不加權


def rank01(v):
    v = np.asarray(v, float)
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def loco(X, y, center):
    out = np.empty(len(y), float)
    for c in np.unique(center):
        te = center == c
        Xtr, Xte = normalize(X[~te]), normalize(X[te])
        sc = StandardScaler().fit(Xtr)
        m = RidgeClassifier(alpha=1.0 / C).fit(sc.transform(Xtr), y[~te])
        out[te] = rank01(m.decision_function(sc.transform(Xte)))
    return out


def ev(y, s):
    r = official_score(y, s, n_iterations=600, rng=np.random.default_rng(0))
    return r["Score"], r["AUROC Full Dataset"], fpr_at_sensitivity(y, s, 0.9)[0]


def main():
    S, y, center = {}, None, None
    for name, tag in TAGS.items():
        f = ROOT / "runs" / f"feats_{tag}.npz"
        if not f.exists():
            print(f"（跳過 {name}：{f.name} 還沒抽完）"); continue
        d = np.load(f, allow_pickle=True)
        X = d["X"]
        if y is None:
            y, center = d["y"].astype(int), d["center"].astype(int)
        S[name] = loco(X, y, center)
        print(f"載入 {name:<12} dim={X.shape[1]}")
    print()

    hdr = f"{'組合':<34} {'Score':>7} {'AUROC':>7} {'FPR@90':>7}"
    print(hdr); print("-" * len(hdr))
    rows = []
    names = list(S)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            s = np.mean([rank01(S[k]) for k in combo], axis=0)
            sc, au, fp = ev(y, s)
            rows.append((sc, combo, au, fp, s))
    for sc, combo, au, fp, _ in sorted(rows, key=lambda t: -t[0]):
        print(f"{'+'.join(combo):<34} {sc:>7.4f} {au:>7.4f} {fp:>7.4f}")

    # 最佳組合 vs 最佳單模型：配對 bootstrap，看差異是不是雜訊
    best = max(rows, key=lambda t: t[0])
    solo = max((r for r in rows if len(r[1]) == 1), key=lambda t: t[0])
    if best[1] != solo[1]:
        print(f"\n=== 配對比較：{'+'.join(best[1])} vs {'+'.join(solo[1])} ===")
        cmp = paired_compare(y, best[4], solo[4], n_iterations=600)
        print(f"  中位差異 {cmp['median_diff']:+.4f}  "
              f"95%CI [{cmp['diff_95CI'][0]:+.4f}, {cmp['diff_95CI'][1]:+.4f}]  "
              f"P(集成較好) = {cmp['P(a>b)']:.3f}")


if __name__ == "__main__":
    main()
