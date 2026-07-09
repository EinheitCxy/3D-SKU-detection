FROM harbor-cn.lingmouai.com/asu/pricetag_ocr_recognition:ppocrv5_250617.2
WORKDIR /app

# 安装系统依赖（使用清华镜像源）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY code/ /app/code/
COPY Pi3/ /app/Pi3/
COPY sam3/ /app/sam3/
COPY api.py ./
COPY processor.py ./
COPY static static
COPY vggt-main/ vggt-main/ 

ENV PYTHONPATH=/app/code:app/vggt-main:/app/Pi3:/app/sam3:/app
ENV HF_HOME=/app/Pi3/checkpoints

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
