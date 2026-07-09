# SKU Global ID Pipeline - Docker使用指南

## 概述

这个Docker容器封装了完整的SKU全局ID生成Pipeline，包括：
1. VGGT 3D匹配
2. 检出框去重
3. 生成带global_id的JSON文件

⚠️ 本项目运行SKU匹配需要GPU（NVIDIA驱动 + nvidia-container-toolkit）。镜像基于 CUDA 12.1 运行时，并预装 GPU 版 PyTorch (2.3.1/cu121)。

## 快速开始

### 1. 构建Docker镜像

```bash
docker build -t sku-global-id-pipeline:latest .
```

### 2. 运行Pipeline

#### 方式一：作为工作流中间节点（推荐，GPU）

上游提供 `images/` 与 `skus`（目录或聚合JSON）；本模块产出带 global_id 的 `skus/` 与原图：

```bash
docker run --gpus all \
           -v /path/to/images:/data/images:ro \
           -v /path/to/skus:/data/skus:ro \
           -v /path/to/output:/output \
           sku-global-id-pipeline:latest \
           --images /data/images \
           --skus /data/skus \
           --output /output
```

输出目录结构：

```
output/
├── images/                       # 原图复制（不修改）
├── skus/                         # 与图片同名的 per-image JSON（已加入 global_id）
└── all_images_with_global_id.json# 合并JSON（保持原始结构）
```

#### 方式二：使用docker-compose（GPU）

```bash
# 编辑 docker-compose.yml 中的 volumes 路径（分别挂载 images 与 skus）
docker compose run --gpus all sku-global-id-pipeline --images /data/images --skus /data/skus --output /output

# 或在 Swarm 模式启用 deploy.devices（见 compose 文件注释）
```

## 数据目录结构要求

输入数据支持两种形式：

1) 目录形式（推荐）：

```
/path/to/images/               # 图片（JPG/PNG/BMP/WEBP）
  ├── 1.jpg
  ├── 2.jpg
  └── ...

/path/to/skus/                 # 检测结果JSON（保持与图片顺序/命名对齐）
  ├── 1.json                   # 优先支持数字命名 1.json、2.json ...
  ├── 2.json
  └── ...
  # 若非数字命名，则需与图片同名：<image_stem>.json（例如 1.jpg → 1.json）
```

2) 聚合文件形式：

- `skus.json` 为一个 JSON 数组，长度必须与图片数量一致；每个元素为单张图片的检测结果结构。

## 输出文件

Pipeline执行完成后，会在输出目录生成：

```
output/
└── dedup/
    └── {dataset_name}/
        ├── all_images_with_global_id.json    # 主要输出：带global_id的合并JSON
        ├── 1_dedup.json                      # 去重后的单图JSON
        ├── 2_dedup.json
        ├── global_mapping.json               # 全局ID映射表
        └── with_global_id/                   # 单图带global_id的JSON
            ├── 1_with_gid.json
            ├── 2_with_gid.json
            └── ...
```

## 命令行参数

### 必需参数（仅工作流模式）

- `--images DIR`          图片目录
- `--skus DIR|FILE`       上游 skus 目录（按图片编号或同名JSON），或聚合 JSON 文件

### 可选参数

- `--output PATH`                     输出根目录（默认: /output）
- `--max-image N`                     限制处理的最大图片数量
- `--algorithm {point_tracking,3d,both}`  匹配算法（默认: 3d）
- `--dedup-mode {any,best}`           去重模式（默认: any）
- `--min-hit-ratio FLOAT`             最小命中率阈值（默认: 0.0）
- `--skip-matching`                   跳过3D匹配（高级用法）

## 使用示例

### 示例1：目录 + 目录（推荐）

```bash
docker run --gpus all \
  -v $(pwd)/imdata/floor_display2/images:/data/images:ro \
  -v $(pwd)/imdata/floor_display2/detections_results:/data/skus:ro \
  -v $(pwd)/docker_output:/output \
  sku-global-id-pipeline:latest \
  --images /data/images \
  --skus /data/skus \
  --output /output
```

### 示例2：目录 + 聚合JSON

```bash
docker run --gpus all \
  -v $(pwd)/imdata/floor_display2/images:/data/images:ro \
  -v $(pwd)/skus.json:/data/skus.json:ro \
  -v $(pwd)/docker_output:/output \
  sku-global-id-pipeline:latest \
  --images /data/images \
  --skus /data/skus.json \
  --output /output
```

### 示例3：限制处理图片数量

```bash
docker run --gpus all \
  -v $(pwd)/imdata/floor_display2/images:/data/images:ro \
  -v $(pwd)/imdata/floor_display2/detections_results:/data/skus:ro \
  -v $(pwd)/docker_output:/output \
  sku-global-id-pipeline:latest \
  --images /data/images \
  --skus /data/skus \
  --output /output \
  --max-image 5
```

### 示例4：使用不同的匹配算法和去重模式

```bash
docker run --gpus all \
  -v $(pwd)/imdata/floor_display2/images:/data/images:ro \
  -v $(pwd)/imdata/floor_display2/detections_results:/data/skus:ro \
  -v $(pwd)/docker_output:/output \
  sku-global-id-pipeline:latest \
  --images /data/images \
  --skus /data/skus \
  --output /output \
  --algorithm both \
  --dedup-mode best \
  --min-hit-ratio 0.6
```

## GPU支持

GPU 为必需条件，需：

1. 安装 NVIDIA 驱动 与 nvidia-container-toolkit
2. 使用 `--gpus all` 启动容器（docker run）或 `docker compose run --gpus all`
3. 若使用 Swarm，可在 compose 中启用 `deploy.resources.reservations.devices` 申请 GPU（compose 文件已提供注释示例）

## 故障排查

### 问题1：权限错误

如果遇到输出目录权限问题，可以在运行前创建输出目录：

```bash
mkdir -p docker_output
chmod 777 docker_output
```

### 问题2：内存不足

如果处理大量图片时内存不足，可以：
- 使用`--max-image`参数限制处理数量
- 增加Docker的内存限制

### 问题3：输入结构错误

- 目录形式：确保 `images/` 内有图片、`skus/` 内有与图片对应的 JSON（数字命名或同名）。
- 聚合文件形式：确保 `skus.json` 是一个数组，长度与图片数量一致。

## 开发模式

如果需要在开发模式下运行（挂载代码目录）：

```bash
docker run --gpus all \
  -v $(pwd)/code:/app/code \
  -v $(pwd)/imdata/floor_display2/images:/data/images:ro \
  -v $(pwd)/imdata/floor_display2/detections_results:/data/skus:ro \
  -v $(pwd)/docker_output:/output \
  sku-global-id-pipeline:latest \
  --images /data/images \
  --skus /data/skus \
  --output /output
```

## 日志

Pipeline会输出详细的执行日志，包括：
- 数据集结构检查
- 3D匹配进度
- 去重统计信息
- 最终输出文件路径

## 技术支持

如有问题，请检查：
1. 数据集目录结构是否正确
2. Docker容器日志：`docker logs sku-pipeline`
3. 输出目录权限是否正确
