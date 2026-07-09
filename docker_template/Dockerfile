FROM harbor-cn.lingmouai.com/asu/max-processor-tray-logic:1.0.0
WORKDIR /app

COPY main.py main.py
COPY processor.py processor.py
COPY static static
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]