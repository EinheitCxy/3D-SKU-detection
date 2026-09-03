FROM harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4

WORKDIR /app

RUN find /app -mindepth 1 -maxdepth 1 -exec rm -rf {} +

COPY --from=venv . /app/.venv/
COPY --from=da3_model . /opt/models/da3/
COPY --from=sam3_checkpoint sam3.pt /app/sam3/checkpoints/sam3.pt
COPY --from=app main.py config.yaml /app/
COPY --from=app src /app/src
COPY --from=app utils /app/utils
COPY --from=app api.py processor.py cos_upload.py /app/
COPY --from=app Depth-Anything-3/src /app/Depth-Anything-3/src
COPY --from=app sam3/sam3 /app/sam3/sam3

RUN --mount=type=bind,from=system_debs,target=/tmp/system-debs,ro \
    dpkg -i /tmp/system-debs/*.deb \
    && test -z "$(dpkg --audit)" \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app:/app/Depth-Anything-3/src:/app/sam3 \
    DA3_VENV_PYTHON=/app/.venv/bin/python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DA3_MODEL_PATH=/opt/models/da3/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7

EXPOSE 80

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
