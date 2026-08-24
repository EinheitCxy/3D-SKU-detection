#!/bin/bash
#!/bin/bash
# Batch current DA3-only 3D pipeline over floor_display datasets.
set -euo pipefail

# 优化CUDA显存分配，避免碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/Output}"
cd "$PROJECT_ROOT"

for i in {1..14}; do
    dataset="$PROJECT_ROOT/imdata/floor_display${i}"
    echo "=== floor_display${i} ==="

    echo "[${i}/14] 3D (DA3)..."
    uv run python main.py --mode pipeline --dataset "${dataset}" --algorithm 3d \
      --match_backend da3 --recon_backend da3 --save_root "$SAVE_ROOT"

    echo ""
done

echo "完成！"
