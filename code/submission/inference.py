"""RARE26 演算法容器進入點。

介面契約（來自 Grand Challenge 的 algorithm interface 設定，非程式碼）：
  輸入  /input/images/stacked-barretts-esophagus-endoscopy/  單一堆疊檔（.tiff/.tif/.mha）
        /input/inputs.json  平台產生，列出本次 job 的 socket
  輸出  /output/stacked-neoplastic-lesion-likelihoods.json
        float 陣列，長度＝堆疊張數，值為 neoplasia 機率

容器一次只處理一個 case，所以推論端不得使用任何跨 case 的統計量
（排名、batch 正規化都不行）。見 model/frozen_probe.py。

授權：MIT（見 LICENSE）。本檔為原創實作，未沿用官方 RARE25-Submission
模板的程式碼 —— 該模板為 CC BY-NC 4.0，與挑戰規則要求的 MIT 不相容。
"""
import json
import sys
from glob import glob
from pathlib import Path

import SimpleITK
import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

IMAGE_SLUG = "stacked-barretts-esophagus-endoscopy"
INPUT_SOCKET = "stacked-barretts-esophagus-endoscopy-images"
OUTPUT_FILE = "stacked-neoplastic-lesion-likelihoods.json"


def read_sockets():
    """回傳本次 job 的 input socket slug（排序後的 tuple）。"""
    with open(INPUT_PATH / "inputs.json") as f:
        return tuple(sorted(sv["interface"]["slug"] for sv in json.load(f)))


def read_stack():
    """讀入堆疊影像，回傳 (N, H, W, 3) 的 uint8 陣列。"""
    folder = INPUT_PATH / "images" / IMAGE_SLUG
    files = sorted(sum((glob(str(folder / f"*.{e}")) for e in ("tiff", "tif", "mha")), []))
    if not files:
        raise FileNotFoundError(f"{folder} 下沒有 tiff/tif/mha")
    if len(files) > 1:
        print(f"警告：{folder} 有 {len(files)} 個檔案，只讀 {files[0]}", file=sys.stderr)
    return SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(files[0]))


def write_likelihoods(values):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / OUTPUT_FILE, "w") as f:
        json.dump([float(v) for v in values], f, indent=4)


def main():
    sockets = read_sockets()
    if sockets != (INPUT_SOCKET,):
        raise ValueError(f"未預期的 socket 組合：{sockets}")

    stack = read_stack()
    print(f"輸入堆疊 shape={stack.shape} dtype={stack.dtype}")

    from model.frozen_probe import FrozenProbe
    model = FrozenProbe(RESOURCE_PATH, batch_size=32, hflip_tta=True)
    scores = model.predict(stack)

    if len(scores) != len(stack):
        raise RuntimeError(f"輸出長度 {len(scores)} 與堆疊張數 {len(stack)} 不符")
    write_likelihoods(scores)
    print(f"已輸出 {len(scores)} 個機率值 → {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
