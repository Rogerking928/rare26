"""修正版 LOCO 評估。

原本 compare.py／ensemble.py 用「各中心內轉排名再合併」，
**那是測試時做不到的正規化**——主辦方把 12 家中心混在一起評分，
而且我們拿不到中心標籤。那個做法抹掉了跨中心的分數尺度差異，
正是未見中心會咬人的地方。

這裡並列三種算法：
  per-center  每個留出中心「單獨」評分 —— 最誠實（一個模型 vs 一個未見中心）
              主要判準是 minimax：max(FPR_c1, FPR_c2)，越低越好
  pooled_raw  兩折的原始 decision_function 直接混合 —— 悲觀
              （混了兩個模型的尺度，測試時只有一個模型，所以偏悲觀）
  pooled_rank 舊做法，中心內排名後合併 —— 樂觀且不可實現，只列出來當對照

⚠ 融合方式已從「排名平均」改為「訓練集 z 標準化後平均」。
排名融合需要共同參照集合，但容器一次只處理一個 case（16 張），
在 16 張裡排名 ≠ 在 25,000 張裡排名 —— 每個 batch 尺度不一致，
主辦方混合評分時會被打散。z 標準化用的是訓練集統計量，
逐樣本可算，batch 大小無關，才是容器裡真的做得出來的東西。
（pooled_rank 欄保留只為顯示舊做法灌水多少。）
"""
import itertools, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from scorer import official_score, fpr_at_sensitivity  # noqa: E402
warnings.filterwarnings("ignore")

# ResNet-50 已在四個判準下確認稀釋（minimax 0.6996），不再納入組合
TAGS = {
    "dino224":    "vit_base_patch14_dinov2_224_tta",
    "convnextv2": "convnextv2_tiny_tta",
    "siglip":     "vit_base_patch16_siglip_224_tta",
    "eva02":      "eva02_base_patch14_224_tta",
    "clip":       "vit_base_patch16_clip_224_tta",
}
C = 0.0003


def rank01(v):
    v = np.asarray(v, float)
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def fold_raw(X, y, center):
    """回傳 {中心: (原始 decision_function, 訓練集 z 標準化後的分數)}。

    z 用的是**訓練折**的 mean/std —— 容器裡可以把這兩個數字打包進權重，
    推論時逐樣本套用，與 batch 大小無關。
    """
    out = {}
    for c in np.unique(center):
        te = center == c
        Xtr, Xte = normalize(X[~te]), normalize(X[te])
        sc = StandardScaler().fit(Xtr)
        m = RidgeClassifier(alpha=1.0 / C).fit(sc.transform(Xtr), y[~te])
        dtr = m.decision_function(sc.transform(Xtr))
        dte = m.decision_function(sc.transform(Xte))
        out[c] = (dte, (dte - dtr.mean()) / (dtr.std() + 1e-12))
    return out


def main():
    raws, y, center = {}, None, None
    for name, tag in TAGS.items():
        f = ROOT / "runs" / f"feats_{tag}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if y is None:
            y, center = d["y"].astype(int), d["center"].astype(int)
        raws[name] = fold_raw(d["X"], y, center)

    centers = sorted(np.unique(center))
    hdr = (f"{'組合':<30} | {'c1 FPR':>7} {'c2 FPR':>7} {'minimax':>8} | "
           f"{'c1 Sc':>7} {'c2 Sc':>7} | {'pooled_z':>9} {'舊rank':>8}")
    print(hdr); print("-" * len(hdr))

    rows = []
    names = list(raws)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            per_fpr, per_sc, pooled_raw, pooled_rank = {}, {}, np.empty(len(y)), np.empty(len(y))
            for c in centers:
                te = center == c
                # 折內排名融合（測試時對整個測試集排名，可實現）
                # 可實現的融合：訓練集 z 標準化後平均
                fused = np.mean([raws[k][c][1] for k in combo], axis=0)
                # 舊的、不可實現的做法，只留著當對照
                pooled_rank[te] = np.mean([rank01(raws[k][c][0]) for k in combo], axis=0)
                pooled_raw[te] = fused
                yc = y[te]
                per_fpr[c] = fpr_at_sensitivity(yc, fused, 0.9)[0]
                per_sc[c] = official_score(
                    yc, fused, n_iterations=400,
                    rng=np.random.default_rng(0))["Score"]
            mm = max(per_fpr.values())
            praw = official_score(y, pooled_raw, n_iterations=400,
                                  rng=np.random.default_rng(0))["Score"]
            prank = official_score(y, pooled_rank, n_iterations=400,
                                   rng=np.random.default_rng(0))["Score"]
            rows.append((mm, combo, per_fpr, per_sc, praw, prank))

    for mm, combo, pf, ps, praw, prank in sorted(rows):
        print(f"{'+'.join(combo):<30} | {pf[1]:>7.4f} {pf[2]:>7.4f} {mm:>8.4f} | "
              f"{ps[1]:>7.4f} {ps[2]:>7.4f} | {praw:>9.4f} {prank:>8.4f}")

    print("\n（依 minimax FPR 由小到大排序 —— 大師指定的主要判準）")


if __name__ == "__main__":
    main()
