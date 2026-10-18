# Validation and final review — 2026-10-18

本轮由协调者调度真实video3/GPU2与CPU组件，三个medium worker按文件归属实现，最终转只读复核。完整结果：[pair-summary.json](pair-summary.json)；直接输出比较：[pair-validation.json](pair-validation.json)。每臂一次，不把组件差额相加成全链路收益。

## 已执行的定向测试

Python均通过uv指向既有`runtime/viewer-latency-20261017/test-env/bin/python`，CPU测试隐藏CUDA；未安装依赖。下表最后几组是后续改动所需的最小补测，与先前集合有重叠，不能相加当唯一测试总数。

| 范围 | 结果 | 覆盖与证据 |
|---|---|---|
| `test/test_da3_model_loading.py`最终集合 | 11 passed | 跳过范围、线程/异常恢复、严格加载、缺键/shape/dtype、共享别名和strided-view拒绝；[DA3报告](da3_loading_report.md) |
| `test/test_da3_3d_reconstructor.py test/test_da3_worker.py` | 23 passed | 既有重建器/worker行为 |
| `test/test_web_viewer_export.py -k 'point_ranges or export_publishes_schema3_bundle_with_product_thumbnails'` | 10 passed, 42 deselected | int32/int64、空/缺label区间、删除观察、schema3输出 |
| `test/test_pointcloud_filter.py`初始完整owned集合 | 39 passed | 原保护与guard、全域评价、cap、阈值、退化和随机域；[ground报告](ground_report.md) |
| ground日志/拒绝分支后续focused | 6 passed, 33 deselected | scalar日志与保护/拒绝分支 |
| ground最终默认0/显式启用断言 | 1 passed | 最终缺省行为 |
| mesh/GLB/LOD初始owned集合 | 89 passed | 原导出、几何oracle、RGBA与boundedFIFO；[bake报告](bake_report.md) |
| projection mode/frame constants/report补测 | 7 passed, 76 deselected | 显式模式与常量复用 |
| 最终default2/非法mode/FIFO补测 | 6 passed, 33 deselected | default2、非法mode不创建目录、显式1/2/4顺序与输出 |

没有运行无关的用户camera测试或全量pytest；生产源码冻结在candidate快照后才执行最终完整pair，计时后没有再修改生产文件。

## 真实实验与失败门槛

- DA3真实checkpoint：1377物理参数全部覆盖，6省略state名是共享LayerNorm别名。最初key-only audit拒绝记录保留，已纠正根因。严格原生加载、原buffer初始化保留。
- Point ranges：同一2402589点/711观察的独立CPU比较，所有端点精确相等；完整pair的metadata也精确相等。
- Ground50K：同一3425675点域10.879015→.980081s，但背景mask变化/plane方向不支持默认推广；保持cap0，实验实现保留。最初readonly mmap→Open3D失败发生在RANSAC前，记录保留，不计性能样本。
- Camera-basis：真实29帧样本有6alpha texel变化，共同不透明RGB最大差1；默认source_world保留原运算次序。合成12例通过不能覆盖该真实反例。
- Atlas1/2/4：同一generation完整导出和所有geometry/128295600decodedRGBA texels相等，选择默认2。单轮固定顺序不足以证明4稳定额外收益。
- 最终GPU2 pair：15项DA3数值数组、mapping、8Viewer arrays、objects/manifest和最终GLB整个payload精确相等。任务200ms采样峰两臂均14.777GiB；完整253.139→186.894s。未做hash/CRC。

## 三位medium独立最终复核

| Reviewer | 只读范围 | 结论 |
|---|---|---|
| profile_gpu_map | checkpoint/同次GPU构造与加载/CPU组件区别/15数组/显存口径 | PASS |
| profile_geometry_map | 前缀降幅/point ranges/Viewer输出/ground不推广 | PASS |
| profile_delivery_map | GLB full pair/累计encode与join重叠/1-2-4组件/默认模式 | PASS |

三者核对最终JSON与canonical文档§12，未自行启动额外GPU或重复测试。历史六视频均值不被本轮单video pair替换。未验收多视频优化后均值/P95、外部准确率、人工画质、热服务、HTTP/COS或浏览器。

## 收尾检查

协调者在最终文档写完后运行一次本轮Python AST解析、本报告与canonical文档本地链接存在性检查、`git diff --check`；结果写入[completion.json](completion.json)。这些静态检查不替代上述已执行的功能测试与真实输出验证。原GPU2 resident及其他用户改动保留，无commit/push/发布/部署。
