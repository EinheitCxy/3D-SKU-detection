# 已有 PyTorch 算子能替代哪些重计算

本轮只调查，不增加GPU实现或新依赖。耗时是此前六视频同次运行的均值，不能与本轮video3组件/组合对照混加。最终是否更快必须包含H2D/D2H、CUDA初始化和导出收尾。

| 方向 | 现有证据 | 可用的已有算子 | 判断 |
|---|---|---|---|
| RGB采样与延展作为一个有界GPU阶段 | bilinear函数14.333s、RGB延展20.619s，分别包含在对应bake父块中 | `torch.nn.functional.grid_sample`、gather、布尔mask/整数赋值 | 优先研究；按frame保持纹理/depth常驻，批1024，不应每批重复搬图或初始化CUDA |
| texel投影、深度有效性与RGB一起计算 | texel投影10.704s，depth/source4.816s | batched matmul、广播、gather、比较 | 与上一项融合才可能抵消搬运/调度成本；只迁小矩阵乘法优先级低 |
| NN/SOR并行准备 | native NN3.326s、SOR5.736s；已有记录支持调查GIL限制PVS构造重叠 | 保留Open3D算法，先评估进程隔离/共享只读点数组 | 比直接重写邻域算法更容易保留语义，但这不是一个现成Torch替换；复制与PCD重建也需计时 |
| DBSCAN | native均值32.259s | Torch核心没有直接等价的空间索引DBSCAN入口 | 不用全量`torch.cdist(N,N)`替代。4百万点float32距离矩阵约64TB，分块只降低峰值而不改变O(N²)工作量；代理/新空间索引另做正确性实验 |
| 自适应预算/稳定排序 | feature约0.436s、预算约2.520s | sort、bincount、cumsum等 | 较低优先级；稳定tie、float64概率、分组循环及GPU搬运可能抵消收益 |

## 必须保留的数值细节

`src/surfel_mesh.py` 的双线性坐标有clip和half-pixel映射。若用`grid_sample`，需明确`align_corners=False`与border行为，并验证边界；当前NumPy坐标与intp相减可提升为float64，直接GPU float32可能改变uint8取整。RGBA延展每轮读取旧known，并按固定四方向先到先得；不能换成无方向max-pool或任意插值。alpha保持二值，原深度/来源比较和Surfel最终直径限制不变。

当前camera-basis CPU实验已出现少量真实alpha变化，说明只在合成平面上相等不足以证明真实像素等价。后续GPU实现同样需要source JPEG、f16几何、真实深度边界和最终palette后的比较，不能把数学等价当成逐位相同。

## 最小下一步

锁定一个真实Viewer generation，按frame加载一次纹理和depth；先实现一个bounded RGB+mask+延展候选，只用现有Torch依赖。记录完整stage含两向搬运和context启动时间，以及最终RGBA/alpha/geometry差异。没有这项对照之前，不给GPU加速倍数或默认推广建议。
