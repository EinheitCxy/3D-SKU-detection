# da3 SKU Matching 自动优化 - 循环状态

## Current Best
- da3 Recall 71.5% / Precision 85.5% (Initial baseline, 2026-07-16, commit 3635c8f)

## Queue (Planner 候选, 按预期收益排序)
1. Top-3 几何重排序（best_match 保留 Top-3 候选，质心距离定夺）
2. 评分权重调整（投影命中率 0.5->0.6）
3. SAM3 采样改进（mask 质量过滤）
4. pairing_3d=all + 唯一性全局排序
5. 多帧可见性投票
6. 平面残差 gating

## Cycle Log
(待首 cycle)

## Notes
- 已修: 尺寸bug / backend-aware阈值 / 删depth_consistency / plane0.2评分 / image_ids修复
- 已知漏检分布: fd2(34%)/fd12(52%) drift重, fd6(87%) 表现好
- SAM3 采样点 vs bbox全点投影分布不同（诊断发现）
