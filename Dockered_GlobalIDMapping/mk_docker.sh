#!/bin/bash
set -e
service=global-id-mapping
edition=1.0.0

# 构建镜像
echo "构建Docker镜像..."
docker build -t harbor-cn.lingmouai.com/asu/$service:$edition .
docker tag harbor-cn.lingmouai.com/asu/$service:$edition harbor-cn.lingmouai.com/asu/$service:latest
echo "构建完成！请运行"

docker run --rm --name global-id-mapping -p 8000:8000 \
  --gpus all \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  harbor-cn.lingmouai.com/asu/$service:$edition
# #   --gpus '"device=1"' \
docker push harbor-cn.lingmouai.com/asu/$service:$edition
