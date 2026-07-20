# da3 vs pi3 全量准确率对比 (floor_display2..12)

生成时间: 2026-07-17 (Cycle3 唯一性 fallback 分配后, commit 255f4f2; pi3 仍为旧值)

> da3 经 Cycle3 优化: 唯一性 fallback 分配(救竞争淘汰漏检)。pi3 未重跑, 保持历史值作对比基准。


## 逐数据集对比

| 数据集 | da3 Recall | da3 Precision | pi3 Recall | pi3 Precision | da3 pred | pi3 pred |
|---|---|---|---|---|---|---|
| floor_display2 | 34.0% | 45.9% | 36.0% | 43.5% | 106 | 217 |
| floor_display3 | 85.4% | 91.1% | 89.6% | 94.5% | 90 | 111 |
| floor_display4 | 71.4% | 90.2% | 77.3% | 90.8% | 122 | 205 |
| floor_display5 | 83.3% | 90.5% | 68.4% | 85.7% | 105 | 110 |
| floor_display6 | 87.9% | 96.1% | 52.0% | 67.3% | 179 | 186 |
| floor_display7 | 82.1% | 94.4% | 82.9% | 94.8% | 305 | 335 |
| floor_display8 | 75.9% | 87.5% | 74.5% | 85.1% | 120 | 150 |
| floor_display9 | 77.5% | 91.2% | 76.5% | 93.1% | 181 | 224 |
| floor_display10 | 86.8% | 95.2% | 91.2% | 96.9% | 62 | 71 |
| floor_display11 | 80.6% | 92.9% | 72.1% | 86.9% | 224 | 234 |
| floor_display12 | 56.8% | 65.8% | 47.1% | 62.1% | 305 | 388 |
| **TOTAL** | **73.3%** | **85.1%** | **67.4%** | **81.1%** | 1809 | 2231 |

## 总体汇总

- **da3** (Cycle3): Recall 73.3% (1543/2106), Precision 85.1% (1535/1804) - commit 255f4f2
- **pi3**: Recall 67.4% (1418/2105), Precision 81.1% (1410/1738) - 历史值,未重跑
- da3 相对 pi3: Recall +5.9%, Precision +4.0%

## da3 提升轨迹

da3 全量 Recall: 22.8%(尺寸bug) → 29.6%(image_ids错位暴露) → 71.5%(image_ids修复) → 71.5%(当前)

## 逐数据集 da3 详细 (TP/GT/预测数/Ref数)

| 数据集 | Ref数 | TP | GT | 预测数 | Recall | Precision |
|---|---|---|---|---|---|---|
| floor_display2 | 4 | 51 | 150 | 190 | 34.0% | 47.7% |
| floor_display3 | 4 | 81 | 96 | 113 | 84.4% | 91.0% |
| floor_display4 | 11 | 109 | 154 | 176 | 70.8% | 90.8% |
| floor_display5 | 8 | 92 | 114 | 113 | 80.7% | 91.1% |
| floor_display6 | 9 | 172 | 198 | 185 | 86.9% | 96.0% |
| floor_display7 | 16 | 288 | 351 | 327 | 82.1% | 94.1% |
| floor_display8 | 8 | 108 | 145 | 147 | 74.5% | 88.0% |
| floor_display9 | 8 | 156 | 213 | 214 | 73.2% | 92.9% |
| floor_display10 | 7 | 59 | 68 | 66 | 86.8% | 96.7% |
| floor_display11 | 10 | 203 | 258 | 244 | 78.7% | 92.7% |
| floor_display12 | 14 | 186 | 359 | 398 | 51.8% | 65.0% |