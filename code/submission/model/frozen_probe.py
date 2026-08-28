"""凍結骨幹集成 + 線性頭的推論器。

與訓練端 code/analysis/export_ensemble.py 逐步對齊：
  壓扁 resize → 骨幹 → L2 正規化 → 標準化 → ridge → 訓練集 z 標準化
  → 跨成員平均 → sigmoid

兩個關鍵設計，都是為了「容器一次只看一個 case（16 張）」這件事：
  1. 融合用 z 標準化而非排名平均 —— 排名需要共同參照集合，16 張裡排名
     不等於 25,000 張裡排名，會讓各 batch 的輸出尺度不一致。
  2. 所有統計量（feat_mean/scale、d_mean/d_std）都來自訓練集，打包在 json 裡，
     推論時逐樣本套用。
預處理是**直接壓扁成方形**，不是 timm 預設的 Resize+CenterCrop（見 NOTES.md）。

**骨幹與成員是多對一。** 一個 ViT 骨幹可以同時供應 CLS 與 mean-pool 兩個成員，
共用同一次前向 —— 這是 mean-pool 那個軸「零額外推論成本」的來源。
成員的 feat 欄位：pool（timm 預設的池化輸出，預設值）／cls／mean。
"""
import json

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T


class FrozenProbe:
    def __init__(self, resource_dir, batch_size=32, hflip_tta=True, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.hflip_tta = hflip_tta

        cfg = json.loads((resource_dir / "ensemble.json").read_text(encoding="utf-8"))
        backbones = cfg.get("backbones")
        if backbones is None:                      # 舊格式：一成員一骨幹
            backbones = [{"model_name": m["model_name"], "img_size": m.get("img_size"),
                          "input_size": m["input_size"], "norm_mean": m["norm_mean"],
                          "norm_std": m["norm_std"]} for m in cfg["members"]]
            for i, m in enumerate(cfg["members"]):
                m["backbone"] = i

        self.backbones = []
        for i, b in enumerate(backbones):
            kw = {"img_size": b["img_size"]} if b.get("img_size") else {}
            net = timm.create_model(b["model_name"], pretrained=False,
                                    num_classes=0, **kw)
            net.load_state_dict(torch.load(resource_dir / f"backbone_{i}.pth",
                                           map_location="cpu"))
            net.to(self.device).eval()
            h, w = b["input_size"]
            self.backbones.append({
                "net": net,
                "npre": getattr(net, "num_prefix_tokens", 1),
                "tf": T.Compose([T.Resize((h, w)), T.ToTensor(),
                                 T.Normalize(mean=b["norm_mean"], std=b["norm_std"])]),
            })

        self.members = []
        for m in cfg["members"]:
            self.members.append({
                "backbone": m["backbone"], "feat": m.get("feat", "pool"),
                "feat_mean": np.asarray(m["feat_mean"], np.float32),
                "feat_scale": np.asarray(m["feat_scale"], np.float32),
                "coef": np.asarray(m["coef"], np.float32),
                "intercept": m["intercept"],
                "d_mean": m["d_mean"], "d_std": m["d_std"], "name": m["name"],
            })
        print(f"FrozenProbe: {len(self.members)} 成員 / {len(self.backbones)} 骨幹 "
              f"({', '.join(m['name'] for m in self.members)}) "
              f"device={self.device} tta={hflip_tta}")

    @torch.no_grad()
    def _forward(self, bb, xb, kinds):
        """一次前向，回傳 {kind: 特徵}。kinds ⊆ {pool, cls, mean}。"""
        net, npre = bb["net"], bb["npre"]
        if kinds == {"pool"}:
            return {"pool": net(xb)}
        t = net.forward_features(xb)               # ViT: [B, npre+N, D]
        out = {}
        if "cls" in kinds:
            out["cls"] = t[:, 0]
        if "mean" in kinds:
            out["mean"] = t[:, npre:].mean(1)
        if "pool" in kinds:
            out["pool"] = net.forward_head(t)
        return out

    @torch.no_grad()
    def _features(self, bi, kinds, imgs):
        bb = self.backbones[bi]
        acc = {k: [] for k in kinds}
        for i in range(0, len(imgs), self.batch_size):
            xb = torch.stack([bb["tf"](im) for im in imgs[i:i + self.batch_size]]
                             ).to(self.device)
            f = self._forward(bb, xb, kinds)
            if self.hflip_tta:
                g = self._forward(bb, torch.flip(xb, dims=[3]), kinds)
                f = {k: (f[k] + g[k]) / 2 for k in kinds}
            for k in kinds:
                acc[k].append(f[k].float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in acc.items()}

    def predict(self, images):
        pil = [Image.fromarray(np.asarray(im)).convert("RGB") for im in images]
        need = {}
        for m in self.members:
            need.setdefault(m["backbone"], set()).add(m["feat"])
        feats = {bi: self._features(bi, kinds, pil) for bi, kinds in need.items()}

        zs = []
        for m in self.members:
            X = feats[m["backbone"]][m["feat"]].copy()
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)   # L2 正規化
            Xs = (X - m["feat_mean"]) / m["feat_scale"]
            d = Xs @ m["coef"] + m["intercept"]
            zs.append((d - m["d_mean"]) / (m["d_std"] + 1e-12))
        z = np.mean(zs, axis=0)
        return [float(v) for v in 1.0 / (1.0 + np.exp(-z))]
