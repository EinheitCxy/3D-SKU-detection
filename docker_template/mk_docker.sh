#!/bin/bash
service=global_id_mapping
edition=0.0.2
docker build -t harbor-cn.lingmouai.com/test/$service:$edition .
docker tag harbor-cn.lingmouai.com/test/$service:$edition harbor-cn.lingmouai.com/test/$service:latest
# docker push harbor-cn.lingmouai.com/asu/$service:$edition

#docker run -p 9001:80 -v /Users/ho/www/omni/www/max-processor-tray-logic:/app $service:$edition
