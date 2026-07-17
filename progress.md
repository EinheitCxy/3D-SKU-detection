# da3 SKU Matching 自动优化 - 循环状态

## 最终成果 (Rick 确认接受)
- **da3 Recall 73.27% / Precision 85.09% (1543/2106 TP, 1804 common)** - Cycle 3 KEEP, commit 255f4f2
- Initial baseline: R71.50% / P85.54% (1505/2106, 1750 common), commit 7e80008
- **净提升: Recall +1.77pt, Precision -0.45pt (可接受)**
- 关键发现: da3 整体重建质量优于 pi3 (da3 R71.5% > pi3 R67.4%), 非此前假设的"da3更差"。R73.27% 是当前架构合理上限。

## Cycle Log (7 cycle/诊断, 1 成功)
- C1 [discard] gaussian bug: 采样分布非主因
- C2 [discard] GT-free SE2: GT-free估计密集货架失败
- C3 [KEEP] 唯一性fallback: R+1.77pt 救竞争淘汰 committed 255f4f2 ✅唯一成功
- C4c [discard] mask回退non-overlap: fd12 P崩产+651冗余FP
- C5 [discard-abtest] 禁用SAM3: fd6 R-2.02pt SAM3保密集覆盖有正价值
- C6 [discard-diag] 外参诊断: 反转前提(da3自洽68px<pi3 92px,da3 9/11>=pi3),外参修正无增益

## 关键教训 (写入未来记忆)
1. **离线复现≠pipeline质量** (C4c): 投影命中GT框 ≠ 经3D验证+唯一性后仍正确
2. **单点投影精度指标误导** (C5): top1==GT率不反映pipeline TP(取决于采样覆盖非仅top1)
3. **局部现象勿推全局** (C6): "pi3 2-3x优da3"是per-ref错框率非全局R,da3整体实优pi3
4. **GT-free估计在密集场景根本不可行** (C2): best-box信号指向各自最近框方向散乱

## 已优化 (Cycle 3 唯一成功改动)
- geometry_3d.py find_best_matching_bbox_with_3d_validation: 返回 validated_candidates 列表
- geometry_3d.py apply_uniqueness_constraint: 贪心 fallback 分配(次优非冲突框救竞争淘汰)

## Ruled Out (全部实证否决)
- pairing next->all (Rick, 计算超预算)
- GT-free SE2漂移 (C2, GT-free估计失败)
- gap门控/score权重 (诊断, FP高质量不可分)
- max_3d_validation_candidates提升 (诊断, +0.15pp)
- Top-3 rerank (C3覆盖)
- 空mask回退non-overlap (C4c, 产海量FP)
- 全局禁用SAM3 mask (C5 A/B, R-2pt)
- da3外参修正 (C6, 无系统性偏差可修且伤da3强数据集)

## 未采纳的残留边际方向 (Rick 决定不再探索)
- 深度去噪(bilateral/median): R+1-3%边际,低风险,但da3已优pi3
- matching层混合路由: 净增益极小(最好回到SAM3水平)

## Notes
- Current best: R73.27%/P85.09% commit 255f4f2 (production code clean, 仅C3 commit)
- da3 整体优于 pi3, R73.27% 是合理架构上限
- 循环按 program.md 协议完整运行至 Rick 确认接受成果
