# RARE26 演算法容器。原創 Dockerfile（MIT，見 LICENSE）。
#
# 基底用 python-slim + CPU-only torch wheel，不用 pytorch CUDA 映像：
# 平台時限是 600 秒/case，實測每 job p95 35.8 秒（2 vCPU 綁定量測），
# CPU 已有 16.6 倍餘裕。CUDA 映像會讓 image 從約 0.8 GB 膨脹到 10 GB 以上，
# 上傳與匯入都變慢，而且本機無 GPU 無法驗證那條路徑。
# FrozenProbe 本身會自動偵測 cuda，日後要換 GPU 不需改程式。

FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/app/.local/bin:${PATH}"

# 平台以非 root 執行容器；先建立使用者再安裝，讓套件落在該使用者的 --user 目錄
RUN useradd --create-home --shell /usr/sbin/nologin app
USER app
WORKDIR /opt/app

COPY --chown=app:app requirements.txt ./
RUN python -m pip install --user --requirement requirements.txt

COPY --chown=app:app resources ./resources
COPY --chown=app:app model ./model
COPY --chown=app:app inference.py ./

ENTRYPOINT ["python", "inference.py"]
