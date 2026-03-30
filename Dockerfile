FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先装 torch CPU 版本
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

# 强制 numpy 1.24.4（与 torch 2.3.0 兼容）
RUN pip install --no-cache-dir "numpy==1.24.4" --force-reinstall

# 装 timm
RUN pip install --no-cache-dir timm

# 装 MobileSAM
RUN pip install --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# 装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再次强制 numpy 版本，防止被覆盖
RUN pip install --no-cache-dir "numpy==1.24.4" --force-reinstall

COPY main.py .

CMD ["python", "main.py"]
