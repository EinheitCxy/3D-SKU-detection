FROM harbor-cn.lingmouai.com/asu/pricetag_ocr_recognition:ppocrv5_250617.2
WORKDIR /app

# 安装OpenCV系统依赖（使用清华镜像源）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制应用文件
COPY main.py main.py
COPY processor.py processor.py
COPY static static

# 复制依赖文件并安装
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制VGGT模块（通过符号链接）
COPY vggt-main/ /app/vggt-main/
COPY code/modules/ /app/code/modules/
COPY code/utils/ /app/code/utils/
COPY code/viewer/ /app/code/viewer/
COPY code/config.yaml app/code/config.yaml

# 设置Python路径
ENV PYTHONPATH=/app:/app/code:/app/vggt-main

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]