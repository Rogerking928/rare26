"""在全部訓練資料上訓練線性頭，連同骨幹權重打包進 submission/resources/。

刻意與 probe.py 用同一組 fit 程式碼路徑（StandardScaler + LogisticRegression），
避免「本機驗證用 A、送出去的是 B」這種對不上的錯。
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "code" / "submission" / "resources"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="runs/feats_<tag>.npz")
    ap.add_argument("--model", required=True, help="timm 模型名，須與抽特徵時相同")
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--img-size", type=int, default=0,
                    help="須與抽特徵時相同；DINOv2 預設 518，覆寫成 224")
    args = ap.parse_args()

    d = np.load(ROOT / "runs" / f"feats_{args.tag}.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    print(f"訓練頭：X={X.shape} 正樣本={y.sum()} C={args.C} balanced={args.balanced}")

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(
        C=args.C, max_iter=5000,
        class_weight="balanced" if args.balanced else None,
    ).fit(sc.transform(X), y)

    kw = {"img_size": args.img_size} if args.img_size else {}
    model = timm.create_model(args.model, pretrained=True, num_classes=0, **kw)
    cfg = timm.data.resolve_data_config({}, model=model)
    if args.img_size:
        cfg["input_size"] = (cfg["input_size"][0], args.img_size, args.img_size)
    RES.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RES / "backbone.pth")

    meta = {
        "model_name": args.model,
        "input_size": list(cfg["input_size"][1:]),
        "norm_mean": list(map(float, cfg["mean"])),
        "norm_std": list(map(float, cfg["std"])),
        "C": args.C,
        "balanced": args.balanced,
        "feat_tag": args.tag,
        "img_size": args.img_size or None,
    }
    np.savez(RES / "head.npz", mean=sc.mean_, scale=sc.scale_,
             coef=clf.coef_, intercept=clf.intercept_[0],
             meta=json.dumps(meta))

    print("已寫入:")
    for f in ("backbone.pth", "head.npz"):
        print(f"  resources/{f}  {(RES/f).stat().st_size/1e6:.1f} MB")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
