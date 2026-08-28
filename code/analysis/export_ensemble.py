"""把選定的骨幹組合訓練成最終模型，打包進 submission/resources/。

流程與 evaluate2.py 逐步對齊，不可有任何差異：
  L2 正規化特徵 → StandardScaler → RidgeClassifier.decision_function
  → 用**訓練集**的 mean/std 做 z 標準化 → 跨成員平均 → sigmoid

z 標準化的統計量來自訓練集，所以推論時逐樣本可算，與 batch 大小無關 ——
容器一次只看 16 張，排名融合在那裡做不出來（見 NOTES.md）。
"""
import argparse, json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "code" / "submission" / "resources"

# name -> (feats tag, timm 模型名, img_size 覆寫)
MEMBERS = {
    "dino224":    ("vit_base_patch14_dinov2_224_tta", "vit_base_patch14_dinov2.lvd142m", 224),
    "convnextv2": ("convnextv2_tiny_tta", "convnextv2_tiny.fcmae_ft_in22k_in1k", 0),
    "siglip":     ("vit_base_patch16_siglip_224_tta", "vit_base_patch16_siglip_224.webli", 0),
    "eva02":      ("eva02_base_patch14_224_tta", "eva02_base_patch14_224.mim_in22k", 0),
    "clip":       ("vit_base_patch16_clip_224_tta", "vit_base_patch16_clip_224.openai", 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, choices=list(MEMBERS))
    ap.add_argument("--C", type=float, default=0.0003)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    for f in RES.glob("*"):
        f.unlink()

    bundle, y_ref = [], None
    for i, name in enumerate(args.members):
        tag, model_name, img_size = MEMBERS[name]
        d = np.load(ROOT / "runs" / f"feats_{tag}.npz", allow_pickle=True)
        X, y = d["X"], d["y"].astype(int)
        y_ref = y if y_ref is None else y_ref
        assert np.array_equal(y, y_ref), "特徵檔的樣本順序不一致"

        Xn = normalize(X)
        sc = StandardScaler().fit(Xn)
        clf = RidgeClassifier(alpha=1.0 / args.C).fit(sc.transform(Xn), y)
        dtr = clf.decision_function(sc.transform(Xn))

        kw = {"img_size": img_size} if img_size else {}
        m = timm.create_model(model_name, pretrained=True, num_classes=0, **kw)
        cfg = timm.data.resolve_data_config({}, model=m)
        h = img_size or cfg["input_size"][1]
        torch.save(m.state_dict(), RES / f"backbone_{i}.pth")

        bundle.append({
            "name": name, "model_name": model_name,
            "img_size": img_size or None, "input_size": [h, h],
            "norm_mean": list(map(float, cfg["mean"])),
            "norm_std": list(map(float, cfg["std"])),
            "feat_mean": sc.mean_.tolist(), "feat_scale": sc.scale_.tolist(),
            "coef": clf.coef_.ravel().tolist(), "intercept": float(clf.intercept_[0]),
            "d_mean": float(dtr.mean()), "d_std": float(dtr.std()),
        })
        print(f"  [{i}] {name:<11} dim={X.shape[1]:>5} {h}x{h}  "
              f"backbone_{i}.pth {(RES/f'backbone_{i}.pth').stat().st_size/1e6:.0f} MB")

    (RES / "ensemble.json").write_text(json.dumps(
        {"C": args.C, "members": bundle}), encoding="utf-8")
    total = sum(f.stat().st_size for f in RES.glob("*")) / 1e6
    print(f"\nresources 共 {total:.0f} MB，成員 {len(bundle)} 個")


if __name__ == "__main__":
    main()
