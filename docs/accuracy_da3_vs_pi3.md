# da3 vs pi3 全量准确率对比 (floor_display2..12)

生成时间: 2026-07-16 (所有改进后: 尺寸修复+backend-aware阈值+删depth_consistency+plane0.2评分+image_ids修复)


## 逐数据集对比

| 数据集 | da3 Recall | da3 Precision | pi3 Recall | pi3 Precision | da3 pred | pi3 pred |
|---|---|---|---|---|---|---|
| floor_display2 | 34.0% | 47.7% | 36.0% | 43.5% | 190 | 217 |
| floor_display3 | 84.4% | 91.0% | 89.6% | 94.5% | 113 | 111 |
| floor_display4 | 70.8% | 90.8% | 77.3% | 90.8% | 176 | 205 |
| floor_display5 | 80.7% | 91.1% | 68.4% | 85.7% | 113 | 110 |
| floor_display6 | 86.9% | 96.0% | 52.0% | 67.3% | 185 | 186 |
| floor_display7 | 82.1% | 94.1% | 82.9% | 94.8% | 327 | 335 |
| floor_display8 | 74.5% | 88.0% | 74.5% | 85.1% | 147 | 150 |
| floor_display9 | 73.2% | 92.9% | 76.5% | 93.1% | 214 | 224 |
| floor_display10 | 86.8% | 96.7% | 91.2% | 96.9% | 66 | 71 |
| floor_display11 | 78.7% | 92.7% | 72.1% | 86.9% | 244 | 234 |
| floor_display12 | 51.8% | 65.0% | 47.1% | 62.1% | 398 | 388 |
| **TOTAL** | **71.5%** | **85.5%** | **67.4%** | **81.1%** | 2173 | 2231 |

## 总体汇总

- **da3**: Recall 71.5% (1505/2106), Precision 85.5% (1497/1750)
- **pi3**: Recall 67.4% (1418/2105), Precision 81.1% (1410/1738)
- da3 相对 pi3: Recall +4.1%, Precision +4.4%

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