"""GastroNet DINOv2 抽特徵。權重到貨當天只要跑這支，不需要任何判斷。

載入方式（來源：HF `BONS-AI-TUE-AMC/GastroNetDinov2` 的 README）：
  timm 的 vit_base_patch14_dinov2.lvd142m 骨架 + img_size 覆寫
  checkpoint 取 state['teacher']，鍵名去掉 'backbone.' 前綴

幾何與其他成員完全一致：壓扁 Resize((S,S)) + hflip TTA + squash。
輸出 runs/feats_gastronet_<S>_tta.npz，欄位與 extract_feats.py 相同。

用法：
  python3 code/analysis/extract_gastronet.py --ckpt ~/Downloads/gastronet.pth [--img-size 336]
"""
import argparse, time
from pathlib import Path
import numpy as np, timm, torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_feats import build_index, ImgSet

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--img-size", type=int, default=336)
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False,
                          num_classes=0, img_size=a.img_size)
state = torch.load(a.ckpt, map_location="cpu")
for key in ("teacher", "student", "model", "state_dict"):
    if isinstance(state, dict) and key in state:
        state = state[key]; print(f"取 checkpoint['{key}']"); break
state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"載入完成：missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing 前 10:", missing[:10])
if len(missing) > 5:
    raise SystemExit("⚠ missing 太多，鍵名對不上，停下來人工檢查，不要硬跑")
model.eval()

cfg = timm.data.resolve_data_config({}, model=model)
S = a.img_size
tf = T.Compose([T.Resize((S, S)), T.ToTensor(),
                T.Normalize(mean=cfg["mean"], std=cfg["std"])])
paths, y, center = build_index()
dl = DataLoader(ImgSet(paths, tf), batch_size=a.batch, num_workers=4, shuffle=False)

feats, t0, done = [], time.time(), 0
with torch.no_grad():
    for xb in dl:
        f = model(xb); f = (f + model(torch.flip(xb, dims=[3]))) / 2
        feats.append(f.numpy().astype(np.float32)); done += len(xb)
        el = time.time() - t0
        print(f"\r  {done}/{len(paths)}  {el:.0f}s ({done/el:.1f} img/s, "
              f"預估 {el/done*len(paths):.0f}s)", end="", flush=True)

X = np.concatenate(feats)
out = ROOT / "runs" / f"feats_gastronet_{S}_tta.npz"
np.savez_compressed(out, X=X, y=y, center=center,
                    paths=np.array([str(p) for p in paths]))
print(f"\n完成 X={X.shape} → {out.name}")
print(f"\n下一步（只執行，不決策）：python3 code/analysis/gastronet_gate.py --tag gastronet_{S}_tta")
