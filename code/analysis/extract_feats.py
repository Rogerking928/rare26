"""凍結骨幹抽特徵。

為什麼是這條路（不是微調 resnet50）：
  - 只有 158 張正樣本。全參數微調 2,500 萬參數必然對正樣本過擬合，
    而官方 metric 恰好對「最難的那幾張正樣本」極度敏感（見 RESUME.md）。
  - 本機無 GPU。抽一次特徵可反覆試分類器，微調則每次都要重跑。

輸出 runs/feats_<tag>.npz：X(float32, N×D)、y(int8)、center(int8)、paths。
"""
import argparse, time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def build_index():
    """回傳 (paths, y, center)。y=1 為 neo，center 為 1/2。"""
    rows = []
    for c in (1, 2):
        for cls, lab in (("ndbe", 0), ("neo", 1)):
            for p in sorted((DATA / f"center_{c}" / cls).glob("*.png")):
                rows.append((p, lab, c))
    paths, y, center = zip(*rows)
    return list(paths), np.array(y, np.int8), np.array(center, np.int8)


class ImgSet(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50.a1_in1k")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hflip", action="store_true",
                    help="水平翻轉後再抽一次，與原圖平均（TTA）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張，用來計時")
    ap.add_argument("--img-size", type=int, default=0,
                    help="覆寫輸入邊長（ViT 會內插 position embedding）。"
                         "DINOv2 預設 518，CPU 上太慢，通常設 224 或 336。")
    ap.add_argument("--transform", choices=["squash", "timm"], default="squash",
                    help="squash=直接壓扁成方形（與測試集幾何一致，見 NOTES）；"
                         "timm=timm 預設的縮放+中心裁切")
    args = ap.parse_args()
    tag = args.tag or args.model.split(".")[0]

    torch.set_num_threads(max(1, torch.get_num_threads()))
    paths, y, center = build_index()
    if args.limit:
        paths, y, center = paths[: args.limit], y[: args.limit], center[: args.limit]

    kw = {"img_size": args.img_size} if args.img_size else {}
    model = timm.create_model(args.model, pretrained=True, num_classes=0, **kw)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    if args.img_size:
        # resolve_data_config 讀的是 default_cfg（DINOv2 寫 518），
        # 不會反映 create_model(img_size=...) 的覆寫，必須自己蓋掉。
        cfg["input_size"] = (cfg["input_size"][0], args.img_size, args.img_size)
    if args.transform == "squash":
        # 官方測試集是 512x512 方形，而訓練原圖約 640x512。
        # 梯度異向性檢定（NOTES.md）顯示主辦方是「壓扁」不是「裁切」，
        # 所以這裡也必須壓扁 —— timm 預設的 Resize+CenterCrop 會切掉兩側，
        # 產生與測試時不同的幾何。
        from torchvision import transforms as T
        h, w = cfg["input_size"][1:]
        tf = T.Compose([T.Resize((h, w)), T.ToTensor(),
                        T.Normalize(mean=cfg["mean"], std=cfg["std"])])
    else:
        tf = timm.data.create_transform(**cfg, is_training=False)
    print(f"model={args.model}  input={cfg['input_size']}  "
          f"transform={args.transform}  threads={torch.get_num_threads()}")

    dl = DataLoader(ImgSet(paths, tf), batch_size=args.batch,
                    num_workers=args.workers, shuffle=False)

    feats, t0, done = [], time.time(), 0
    with torch.no_grad():
        for xb in dl:
            f = model(xb)
            if args.hflip:
                f = (f + model(torch.flip(xb, dims=[3]))) / 2
            feats.append(f.numpy().astype(np.float32))
            done += len(xb)
            el = time.time() - t0
            print(f"\r  {done}/{len(paths)}  {el:.0f}s  "
                  f"({done/el:.1f} img/s, 預估總計 {el/done*len(paths):.0f}s)",
                  end="", flush=True)
    X = np.concatenate(feats)
    print(f"\n完成：X={X.shape}  {time.time()-t0:.0f}s")

    suffix = (f"_{args.img_size}" if args.img_size else "") \
        + ("_tta" if args.hflip else "") \
        + ("_timmtf" if args.transform == "timm" else "")
    out = ROOT / "runs" / f"feats_{tag}{suffix}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, X=X, y=y, center=center,
                        paths=np.array([str(p) for p in paths]))
    print("存到", out, f"({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
