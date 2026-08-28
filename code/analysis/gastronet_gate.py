"""GastroNet 到貨後的閘門。規則在權重到達**之前**就寫死，當天只執行。

裁示（2026-08-28 修正版）：**不要用「GastroNet 單模是否勝過 DINO 單模」
間接決定 ensemble 組成。** 單模強弱不等於互補價值 —— 一個較弱但誤差互補的
模型仍可能形成更好的 ensemble；反之亦然。要直接檢驗的是 ensemble 本身。

固定程序，不掃描：
  1. 抽 GastroNet 特徵
  2. 算 GastroNet 單模 minimax —— **僅作描述**，不參與任何分支
  3. 固定建立 DINO + GastroNet，權重 50:50
  4. 與現行 baseline（DINO + ConvNeXtV2）做同一組配對 bootstrap
  5. 三者全滿足才取代 baseline：
       minimax 改善
       c1、c2 均不惡化超過 0.01
       P(candidate better) >= 0.70
     否則**完全不用 GastroNet**

三成員版本（DINO + ConvNeXtV2 + GastroNet）**只有預先登記後才測**，
這支程式不會自己去試 —— 那會變成在同一份兩中心驗證集上多抽一次。

0.70 而非 0.60：GastroNet 是新權重檔、img_size=336 的新前處理、
沒跑過的載入程式碼，操作風險是實的（即使 runtime 有 16.6 倍餘裕）。

評估流程與所有成員逐步相同：同樣的 LOCO fold、per-center 評分、
訓練集 z 標準化、同一組配對 bootstrap 索引。不得換捷徑。
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore")
C, ALPHA, B = 0.0003, 1.0 / 0.0003, 2000
BASE = {"dino224": "vit_base_patch14_dinov2_224_tta",
        "convnextv2": "convnextv2_tiny_tta"}


def fpr90(yt, ys):
    fpr, tpr, _ = roc_curve(yt, ys)
    return float(fpr[min(np.searchsorted(tpr, 0.9, side="left"), len(tpr) - 1)])


def fold_z(X, y, center):
    out = {}
    for c in np.unique(center):
        te = center == c
        Xtr, Xte = normalize(X[~te]), normalize(X[te])
        sc = StandardScaler().fit(Xtr)
        m = RidgeClassifier(alpha=ALPHA).fit(sc.transform(Xtr), y[~te])
        dtr, dte = m.decision_function(sc.transform(Xtr)), m.decision_function(sc.transform(Xte))
        out[c] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
    return out


def mm_boot(zs, y, center, idx, centers):
    out = np.empty(B)
    for b in range(B):
        out[b] = max(fpr90(y[center == c][idx[c][b]], zs[c][idx[c][b]]) for c in centers)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="例如 gastronet_336_tta")
    a = ap.parse_args()

    Z, y, center = {}, None, None
    for name, tag in {**BASE, "gastronet": a.tag}.items():
        f = ROOT / "runs" / f"feats_{tag}.npz"
        if not f.exists():
            raise SystemExit(f"缺特徵檔 {f}")
        d = np.load(f, allow_pickle=True)
        if y is None:
            y, center = d["y"].astype(int), d["center"].astype(int)
        Z[name] = fold_z(d["X"], y, center)

    centers = sorted(np.unique(center))
    rng = np.random.default_rng(0)
    idx = {}
    for c in centers:
        yc = y[center == c]
        pos, neg = np.where(yc == 1)[0], np.where(yc == 0)[0]
        idx[c] = [np.concatenate([neg, rng.choice(pos, len(pos), replace=True)])
                  for _ in range(B)]

    def fuse(keys):
        return {c: np.mean([Z[k][c] for k in keys], axis=0) for c in centers}

    def report(label, zs):
        pt = {c: fpr90(y[center == c], zs[c]) for c in centers}
        return label, pt, max(pt.values()), mm_boot(zs, y, center, idx, centers)

    # ---- 步驟 1：單模強度，僅作描述，不參與分支
    _, g_pt, g_mm, _ = report("gastronet", fuse(["gastronet"]))
    _, d_pt, d_mm, _ = report("dino224", fuse(["dino224"]))
    print("【描述性，不參與判定】")
    print(f"  gastronet 單模 minimax {g_mm:.4f} (c1 {g_pt[1]:.4f} / c2 {g_pt[2]:.4f})")
    print(f"  dino224   單模 minimax {d_mm:.4f} (c1 {d_pt[1]:.4f} / c2 {d_pt[2]:.4f})")

    # ---- 步驟 2：唯一的正式候選 = DINO + GastroNet 50:50
    members = ["dino224", "gastronet"]
    _, b_pt, b_mm, b_boot = report("baseline", fuse(["dino224", "convnextv2"]))
    _, c_pt, c_mm, c_boot = report("candidate", fuse(members))
    d_ = c_boot - b_boot
    pbet = float(np.mean(d_ < 0))
    dc = {c: c_pt[c] - b_pt[c] for c in centers}
    print(f"\n【正式比較】候選 = {'+'.join(members)}（50:50，不掃權重）")
    print(f"  baseline  minimax {b_mm:.4f}  c1 {b_pt[1]:.4f}  c2 {b_pt[2]:.4f}")
    print(f"  候選      minimax {c_mm:.4f}  c1 {c_pt[1]:.4f}  c2 {c_pt[2]:.4f}")
    print(f"  Δc1 {dc[1]:+.4f}  Δc2 {dc[2]:+.4f}  Δminimax中位 {np.median(d_):+.4f}  "
          f"P(候選較好) {pbet:.3f}")

    why = []
    if c_mm >= b_mm: why.append("minimax 未改善")
    if dc[1] > 0.01: why.append("c1 惡化>0.01")
    if dc[2] > 0.01: why.append("c2 惡化>0.01")
    if pbet < 0.70: why.append(f"P={pbet:.2f}<0.70")
    if why:
        print("\n判定：不採用（" + "、".join(why) + "）→ **完全不用 GastroNet**，"
              "維持 dino224+convnextv2")
    else:
        print("\n判定：採用 dino224+gastronet 取代 baseline")
    print("\n（三成員版本需預先登記才測，本程式不會自行嘗試）")


if __name__ == "__main__":
    main()
