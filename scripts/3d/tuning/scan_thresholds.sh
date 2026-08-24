#!/bin/bash
# 网格扫描 da3 阈值：max_depth_difference × plane_normal_alignment_threshold
# 用法: bash scan_thresholds.sh [dataset_num]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
FD=${1:-2}
GPU=${GPU:-2}
RESULTS=Output/scan_results
mkdir -p "$RESULTS"
REPORT="$RESULTS/scan_fd${FD}.tsv"
echo -e "max_dd\tplane_thr\tRecall\tPrecision\tTP\tGT" > "$REPORT"

# 网格：max_depth_difference ∈ {0.6,0.8,1.0}, plane_threshold ∈ {0.5,0.6,0.7}
for mdd in 0.6 0.8 1.0; do
  for pthr in 0.5 0.6 0.7; do
    echo "=== max_depth_difference=$mdd plane=$pthr ==="
    rm -rf Output/da3/floor_display${FD}/output_3dmapping_da3 Output/da3/floor_display${FD}/accuracy_evaluation_da3
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run python main.py --mode concise --dataset imdata/floor_display${FD} \
      --algorithm 3d --match_backend da3 --recon_backend da3 --save_root Output/da3 \
      --max_depth_difference $mdd --plane_normal_alignment_threshold $pthr > "$RESULTS/fd${FD}_${mdd}_${pthr}.log" 2>&1
    bash scripts/3d/evaluation/accuracy_evaluation.sh floor_display${FD} --backend da3 --save-root Output/da3 > /dev/null 2>&1
    uv run python -c "
import re
txt=open('Output/da3/floor_display${FD}/accuracy_evaluation_da3/summary.txt').read()
PAT=re.compile(r'总体召回率.*?:\s*([\d.]+)%\s*\((\d+)/(\d+)\)\s*.*?有效率.*?:\s*([\d.]+)%\s*\((\d+)/(\d+)\)\s*.*?映射准确率.*?:\s*([\d.]+)%\s*\((\d+)/(\d+)\)',re.S)
b=PAT.findall(txt)
if not b: print('${mdd}\t${pthr}\tNA\tNA\t0\t0'); exit()
tp=sum(int(x[1]) for x in b);gt=sum(int(x[2]) for x in b);co=sum(int(x[7]) for x in b);cm=sum(int(x[8]) for x in b)
print(f'${mdd}\t${pthr}\t{tp/gt:.1%}\t{co/cm:.1%}\t{tp}\t{gt}')
" >> "$REPORT"
  done
done
echo "扫描完成: $REPORT"
column -t -s$'\t' "$REPORT"
