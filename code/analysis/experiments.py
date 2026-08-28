"""三個預先鎖定的一次性實驗 + 統一的放行閘。

放行規則（兩位大師的交集，取較嚴的那個）：
  1. minimax FPR@90 必須改善
  2. c1 與 c2 都不得惡化超過 0.01
  3. 配對 bootstrap 的 P(對手較好) >= 0.70
     （0.70 而非 0.60：三次比較取最好的一次，需要比單次更高的門檻）
任一條不過就維持 baseline。不掃參數、不事後調門檻。

實驗：
  A  DINO mean-pool 取代 CLS / CLS 與 mean-pool 固定 50:50（只對 DINO，
     ConvNeXtV2 本來就是 global pooling）
  B  hard-negative 加權 ridge，**交叉配適**：hard negative 只能由 OOF 分數定義，
     否則用同一個頭找難負樣本再用同一批資料重訓 = 循環擬合。權重固定 2×。
  C  ridge subbagging，10 heads × 80% 分層子抽樣，等權平均。
     順便記錄各 bag 的 sigma 範圍 —— 若某個 bag 的 sigma 極端，融合會被它主導，
     那時「不是 no-op」的原因是數值問題，不是多樣性。
"""
import sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

C = 0.0003
ALPHA = 1.0 / C
B = 2000
HN_WEIGHT = 2.0
N_BAGS, BAG_FRAC = 10, 0.8

TAGS = {
    "dino224":     "vit_base_patch14_dinov2_224_tta",
    "convnextv2":  "convnextv2_tiny_tta",
    "dino224cls":  "dino224cls_tta",
    "dino224mean": "dino224mean_tta",
}


def load():
    F, y, center = {}, None, None
    for name, tag in TAGS.items():
        f = ROOT / "runs" / f"feats_{tag}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if y is None:
            y, center = d["y"].astype(int), d["center"].astype(int)
        F[name] = d["X"]
    return F, y, center


def fpr90(yt, ys):
    fpr, tpr, _ = roc_curve(yt, ys)
    return float(fpr[min(np.searchsorted(tpr, 0.9, side="left"), len(tpr) - 1)])


# ---------------------------------------------------------------- 三種頭

def head_plain(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr), ytr)
    dtr, dte = m.decision_function(sc.transform(Xtr)), m.decision_function(sc.transform(Xte))
    return (dte - dtr.mean()) / (dtr.std() + 1e-12), {}


def head_hardneg(Xtr, ytr, Xte):
    """hard negative 由 5 折 OOF 分數定義，再用全部訓練資料加權重訓。"""
    oof = np.empty(len(ytr))
    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[itr])
        m = RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr[itr]), ytr[itr])
        oof[ite] = m.decision_function(sc.transform(Xtr[ite]))
    thr = np.quantile(oof[ytr == 1], 0.10)          # 90% 敏感度的操作點
    w = np.ones(len(ytr))
    hard = (ytr == 0) & (oof >= thr)                 # 該操作點下的偽陽性
    w[hard] = HN_WEIGHT
    sc = StandardScaler().fit(Xtr)
    m = RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr), ytr, sample_weight=w)
    dtr, dte = m.decision_function(sc.transform(Xtr)), m.decision_function(sc.transform(Xte))
    return (dte - dtr.mean()) / (dtr.std() + 1e-12), {"n_hard": int(hard.sum()),
                                                      "frac_hard": float(hard.mean())}


def head_bagged(Xtr, ytr, Xte):
    zs, sigmas = [], []
    rng = np.random.default_rng(0)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    for b in range(N_BAGS):
        idx = np.concatenate([
            rng.choice(pos, int(round(BAG_FRAC * len(pos))), replace=False),
            rng.choice(neg, int(round(BAG_FRAC * len(neg))), replace=False)])
        sc = StandardScaler().fit(Xtr[idx])
        m = RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr[idx]), ytr[idx])
        db = m.decision_function(sc.transform(Xtr[idx]))
        dte = m.decision_function(sc.transform(Xte))
        sigmas.append(float(db.std()))
        zs.append((dte - db.mean()) / (db.std() + 1e-12))
    return np.mean(zs, axis=0), {"sigma_min": min(sigmas), "sigma_max": max(sigmas),
                                 "sigma_ratio": max(sigmas) / max(min(sigmas), 1e-12)}


HEADS = {"plain": head_plain, "hardneg": head_hardneg, "bagged": head_bagged}


# ---------------------------------------------------------------- 變體

def variant_scores(F, y, center, members, head="plain", weights=None):
    """members: [(名稱, 特徵鍵)]，weights 為成員權重（None = 等權）。"""
    centers = sorted(np.unique(center))
    info = {}
    out = {}
    for c in centers:
        te = center == c
        zs = []
        for key in members:
            X = F[key]
            Xtr, Xte = normalize(X[~te]), normalize(X[te])
            z, meta = HEADS[head](Xtr, y[~te], Xte)
            zs.append(z)
            for k, v in meta.items():
                info.setdefault(f"c{c}_{key}_{k}", v)
        w = np.ones(len(zs)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        out[c] = np.average(zs, axis=0, weights=w)
    return out, info


def evaluate(scores, y, center, idx):
    centers = sorted(np.unique(center))
    point = {c: fpr90(y[center == c], scores[c]) for c in centers}
    mm = np.empty(B)
    for b in range(B):
        mm[b] = max(fpr90(y[center == c][idx[c][b]], scores[c][idx[c][b]]) for c in centers)
    return point, mm


def main():
    F, y, center = load()
    centers = sorted(np.unique(center))
    print("已載入特徵：", ", ".join(sorted(F)))

    rng = np.random.default_rng(0)
    idx = {}
    for c in centers:
        yc = y[center == c]
        pos, neg = np.where(yc == 1)[0], np.where(yc == 0)[0]
        idx[c] = [np.concatenate([neg, rng.choice(pos, len(pos), replace=True)])
                  for _ in range(B)]

    VARIANTS = [("baseline: dino224+convnextv2", ["dino224", "convnextv2"], "plain", None)]
    if "dino224mean" in F:
        VARIANTS += [
            ("A1 mean-pool 取代 CLS",      ["dino224mean", "convnextv2"], "plain", None),
            ("A2 CLS+mean 50:50 +convnext", ["dino224cls", "dino224mean", "convnextv2"],
             "plain", [0.5, 0.5, 1.0]),
        ]
    VARIANTS += [
        ("B  hard-negative 加權 2x", ["dino224", "convnextv2"], "hardneg", None),
        ("C  subbagging 10x80%",     ["dino224", "convnextv2"], "bagged", None),
    ]

    res = {}
    for name, members, head, w in VARIANTS:
        if any(k not in F for k in members):
            print(f"  跳過 {name}（缺特徵）"); continue
        s, info = variant_scores(F, y, center, members, head, w)
        point, mm = evaluate(s, y, center, idx)
        res[name] = (point, mm, info)
        print(f"  完成 {name}")

    base_name = VARIANTS[0][0]
    bp, bmm, _ = res[base_name]
    print(f"\n{'變體':<32} {'c1':>7} {'c2':>7} {'minimax':>8} | "
          f"{'Δc1':>7} {'Δc2':>7} {'Δmm中位':>8} {'P(較好)':>8} | 判定")
    print("-" * 104)
    print(f"{base_name:<32} {bp[1]:>7.4f} {bp[2]:>7.4f} {max(bp.values()):>8.4f} | "
          f"{'—':>7} {'—':>7} {'—':>8} {'—':>8} | baseline")
    for name, (p, mm, info) in res.items():
        if name == base_name:
            continue
        d = mm - bmm
        pbet = float(np.mean(d < 0))
        dc1, dc2 = p[1] - bp[1], p[2] - bp[2]
        mm_pt, bmm_pt = max(p.values()), max(bp.values())
        ok = (mm_pt < bmm_pt) and (dc1 <= 0.01) and (dc2 <= 0.01) and (pbet >= 0.70)
        why = []
        if mm_pt >= bmm_pt: why.append("minimax 未改善")
        if dc1 > 0.01: why.append("c1 惡化>0.01")
        if dc2 > 0.01: why.append("c2 惡化>0.01")
        if pbet < 0.70: why.append(f"P={pbet:.2f}<0.70")
        print(f"{name:<32} {p[1]:>7.4f} {p[2]:>7.4f} {mm_pt:>8.4f} | "
              f"{dc1:>+7.4f} {dc2:>+7.4f} {np.median(d):>+8.4f} {pbet:>8.3f} | "
              f"{'採用' if ok else '不採用：' + '、'.join(why)}")
        if info:
            print(f"{'':32}   診斷 {info}")

    print("\n閘門：minimax 改善 且 兩中心惡化皆 <=0.01 且 P(較好)>=0.70")


if __name__ == "__main__":
    main()
