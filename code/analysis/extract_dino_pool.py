"""一次前向同時抽 DINOv2 的 CLS 與 mean-pool 特徵。

大師裁示的第 1 個實驗：mean-pool 是唯一「換特徵、但零額外推論成本」的多樣性軸
（同一次 forward_features 就能拿到兩者）。只對 DINO 做 —— ConvNeXtV2 本來就是
global pooling，沒有對應的比較。

幾何與 extract_feats.py 完全一致：壓扁 Resize((224,224)) + hflip TTA。
"""
import time
from pathlib import Path
import numpy as np, timm, torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_feats import build_index, ImgSet
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
MODEL, IMG = "vit_base_patch14_dinov2.lvd142m", 224

paths, y, center = build_index()
model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=IMG).eval()
cfg = timm.data.resolve_data_config({}, model=model)
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                T.Normalize(mean=cfg["mean"], std=cfg["std"])])
npre = model.num_prefix_tokens
print(f"{MODEL} @{IMG}  num_prefix_tokens={npre}  threads={torch.get_num_threads()}")

def both(x):
    t = model.forward_features(x)                 # [B, npre+N, D]
    return t[:, 0], t[:, npre:].mean(1)

dl = DataLoader(ImgSet(paths, tf), batch_size=32, num_workers=4, shuffle=False)
CLS, MP, t0, done = [], [], time.time(), 0
with torch.no_grad():
    for xb in dl:
        c1, m1 = both(xb)
        c2, m2 = both(torch.flip(xb, dims=[3]))
        CLS.append(((c1 + c2) / 2).numpy().astype(np.float32))
        MP.append(((m1 + m2) / 2).numpy().astype(np.float32))
        done += len(xb); el = time.time() - t0
        print(f"\r  {done}/{len(paths)}  {el:.0f}s ({done/el:.1f} img/s, "
              f"預估 {el/done*len(paths):.0f}s)", end="", flush=True)

for name, arr in (("dino224cls", CLS), ("dino224mean", MP)):
    X = np.concatenate(arr)
    out = ROOT / "runs" / f"feats_{name}_tta.npz"
    np.savez_compressed(out, X=X, y=y, center=center,
                        paths=np.array([str(p) for p in paths]))
    print(f"\n{name}: X={X.shape} → {out.name} ({out.stat().st_size/1e6:.0f} MB)")
