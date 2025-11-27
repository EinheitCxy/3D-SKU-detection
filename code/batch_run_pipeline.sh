#!/bin/bash
# 批量运行pipeline：floor_display1-14，point_tracking + pi3 3d算法

cd "$(dirname "$0")"

for i in {1..14}; do
    dataset="../imdata/floor_display${i}"
    echo "=== floor_display${i} ==="

    echo "[${i}/14] Point Tracking..."
    uv run python main.py --mode pipeline --dataset "${dataset}" --algorithm point_tracking

    echo "[${i}/14] 3D (PI3)..."
    uv run python main.py --mode pipeline --dataset "${dataset}" --algorithm 3d --match_backend pi3 --recon_backend pi3

    echo "[${i}/14] 3D (VGGT)..."
    uv run python main.py --mode pipeline --dataset "${dataset}" --algorithm 3d --match_backend vggt --recon_backend vggt

    echo ""
done

echo "完成！"
