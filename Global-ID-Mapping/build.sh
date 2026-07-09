#!/bin/bash
set -e
service=global-id-mapping
edition=3.1.0
dir=asu
# test目录做测试

# 构建镜像
echo "构建Docker镜像..."
docker build -t harbor-cn.lingmouai.com/$dir/$service:$edition .
# docker tag harbor-cn.lingmouai.com/$dir/$service:$edition harbor-cn.lingmouai.com/$dir/$service:latest
echo "构建完成！"

docker run -d --name global-id-mapping \
  -p 8011:80 \
  --gpus all \
  harbor-cn.lingmouai.com/$dir/$service:$edition

# docker rm -f global-id-mapping
# docker logs global-id-mapping

# uv run test_api.py

docker push harbor-cn.lingmouai.com/$dir/$service:$edition
# docker push harbor-cn.lingmouai.com/$dir/$service:latest

# nohup ./build.sh > build.log 2>&1 &