"""在同一套 LOCO 協定下，直接比較「監督式分類」與「盛行率無關（異常偵測）」。

這正是 RARE25 總結論文 §7.3 明講缺、且說「would be particularly valuable」的比較：
  Comparative studies that directly contrast supervised classifiers with
  prevalence-agnostic alternatives under identical evaluation protocols.

所有方法共用：同一組凍結特徵、同一組 LOCO 切分、同一個官方 metric。
差別只在打分方式，所以比較是乾淨的。

異常偵測一律**只看正常樣本（ndbe）**擬合，完全不用 158 張病灶標籤。
"""
import argparse, sys, warnings
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from scorer import official_score, fpr_at_sensitivity  # noqa: E402

warnings.filterwarnings("ignore")


# ----------------------------------------------------------- 打分器
# 每個都是 fit(Xtr, ytr) -> predict(Xte)。異常偵測忽略 ytr 的正樣本。

def sup_logreg(Xtr, ytr, Xte, C=0.001):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000,
                             class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def mahalanobis(Xtr, ytr, Xte, shrink=True):
    """只用正常樣本估計常態分布，測試樣本離中心越遠越可疑。"""
    N = Xtr[ytr == 0]
    sc = StandardScaler().fit(N)
    Ns, Ts = sc.transform(N), sc.transform(Xte)
    cov = LedoitWolf().fit(Ns) if shrink else None
    if shrink:
        return cov.mahalanobis(Ts)
    P = np.linalg.pinv(np.cov(Ns.T))
    d = Ts - Ns.mean(0)
    return np.einsum("ij,jk,ik->i", d, P, d)


def knn_dist(Xtr, ytr, Xte, k=5):
    """到 k 個最近正常樣本的平均餘弦距離。"""
    N = normalize(Xtr[ytr == 0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(N)
    return nn.kneighbors(normalize(Xte))[0].mean(1)


def pca_recon(Xtr, ytr, Xte, q=64):
    """在正常樣本上學低維子空間，重建誤差即異常分數。"""
    N = Xtr[ytr == 0]
    sc = StandardScaler().fit(N)
    p = PCA(n_components=q, random_state=0).fit(sc.transform(N))
    T = sc.transform(Xte)
    return ((T - p.inverse_transform(p.transform(T))) ** 2).sum(1)


def rank01(v):
    v = np.asarray(v, float)
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


# ----------------------------------------------------------- 評估
def loco_scores(X, y, center, fn):
    """每個中心輪流當測試集；各中心內轉成排名再合併（metric 只看排序）。"""
    out = np.empty(len(y), float)
    for c in np.unique(center):
        te = center == c
        out[te] = rank01(fn(X[~te], y[~te], X[te]))
    return out


def ev(y, s, seed=0):
    r = official_score(y, s, n_iterations=400, rng=np.random.default_rng(seed))
    return r["Score"], r["AUROC Full Dataset"], fpr_at_sensitivity(y, s, 0.9)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="resnet50_tta")
    args = ap.parse_args()
    d = np.load(ROOT / "runs" / f"feats_{args.tag}.npz", allow_pickle=True)
    X, y, center = d["X"], d["y"].astype(int), d["center"].astype(int)
    print(f"{args.tag}: X={X.shape} 正樣本={y.sum()}\n")

    methods = {
        "監督 logreg C=1e-3": lambda a, b, c: sup_logreg(a, b, c, 1e-3),
        "監督 logreg C=1e-4": lambda a, b, c: sup_logreg(a, b, c, 1e-4),
        "異常 Mahalanobis":   mahalanobis,
        "異常 kNN k=5":       lambda a, b, c: knn_dist(a, b, c, 5),
        "異常 kNN k=20":      lambda a, b, c: knn_dist(a, b, c, 20),
        "異常 PCA q=64":      lambda a, b, c: pca_recon(a, b, c, 64),
        "異常 PCA q=256":     lambda a, b, c: pca_recon(a, b, c, 256),
    }

    S = {}
    hdr = f"{'方法':<22} {'Score':>8} {'AUROC':>8} {'FPR@90':>8}"
    print(hdr); print("-" * len(hdr))
    for name, fn in methods.items():
        s = loco_scores(X, y, center, fn)
        S[name] = s
        sc, au, fp = ev(y, s)
        print(f"{name:<22} {sc:>8.4f} {au:>8.4f} {fp:>8.4f}")

    # 混合：監督 + 最好的異常分數，等權排名平均
    print("\n--- 混合（排名平均，主辦方點名的 hybrid）---")
    sup = S["監督 logreg C=1e-3"]
    best_an = max((k for k in S if k.startswith("異常")),
                  key=lambda k: ev(y, S[k])[0])
    print(f"（異常端取最佳者：{best_an}）")
    for w in (0.25, 0.5, 0.75):
        h = (1 - w) * rank01(sup) + w * rank01(S[best_an])
        sc, au, fp = ev(y, h)
        print(f"{'混合 w_anom=' + str(w):<22} {sc:>8.4f} {au:>8.4f} {fp:>8.4f}")


if __name__ == "__main__":
    main()
