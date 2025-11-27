#!/usr/bin/env python3
"""
汇总分析所有数据集上三个算法模型的总体表现
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_summary_file(file_path: Path) -> Dict:
    """解析summary.txt文件，提取关键指标"""
    if not file_path.exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取所有参考图片的指标
    metrics = []
    pattern = r'总体召回率 \(Recall\): ([\d.]+)%.*?VGGT有效率 \(Effectiveness\): ([\d.]+)%.*?Reference ID映射准确率 \(Precision\): ([\d.]+)%'

    for match in re.finditer(pattern, content, re.DOTALL):
        recall = float(match.group(1))
        effectiveness = float(match.group(2))
        precision = float(match.group(3))
        metrics.append({
            'recall': recall,
            'effectiveness': effectiveness,
            'precision': precision
        })

    if not metrics:
        return None

    # 计算平均值
    avg_recall = sum(m['recall'] for m in metrics) / len(metrics)
    avg_effectiveness = sum(m['effectiveness'] for m in metrics) / len(metrics)
    avg_precision = sum(m['precision'] for m in metrics) / len(metrics)

    return {
        'count': len(metrics),
        'avg_recall': avg_recall,
        'avg_effectiveness': avg_effectiveness,
        'avg_precision': avg_precision,
        'metrics': metrics
    }

def main():
    output_dir = Path(__file__).parent / "Output"

    # 算法名称映射
    algo_mapping = {
        'accuracy_evaluation_pt': 'Point Tracking',
        'accuracy_evaluation_pt_vggt': 'Point Tracking (VGGT)',
        'accuracy_evaluation_3d_vggt': '3D Mapping (VGGT)',
        'accuracy_evaluation_3d_pi3': '3D Mapping (Pi3)',
        'accuracy_evaluation_pi3': '3D Mapping (Pi3)',
        'accuracy_evaluation_vggt': '3D Mapping (VGGT)',
    }

    # 收集所有数据
    all_data = defaultdict(lambda: defaultdict(dict))

    # 扫描code/Output目录
    for dataset_dir in sorted(output_dir.glob("floor_display*")):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # 扫描所有accuracy_evaluation目录
        for eval_dir in dataset_dir.glob("accuracy_evaluation_*"):
            if not eval_dir.is_dir():
                continue

            summary_file = eval_dir / "summary.txt"
            algo_key = eval_dir.name

            # 标准化算法名称
            algo_name = algo_mapping.get(algo_key, algo_key)

            # 解析summary文件
            metrics = parse_summary_file(summary_file)
            if metrics:
                all_data[dataset_name][algo_name] = metrics

    # 生成汇总报告
    report_path = output_dir / "overall_model_performance.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SKU匹配算法总体性能汇总报告\n")
        f.write("=" * 80 + "\n\n")

        # 按数据集展示
        f.write("## 各数据集详细表现\n\n")
        for dataset_name in sorted(all_data.keys()):
            f.write(f"### {dataset_name}\n")
            f.write("-" * 80 + "\n")

            for algo_name in sorted(all_data[dataset_name].keys()):
                metrics = all_data[dataset_name][algo_name]
                f.write(f"\n**{algo_name}**\n")
                f.write(f"  - 评估图片对数: {metrics['count']}\n")
                f.write(f"  - 平均召回率 (Recall): {metrics['avg_recall']:.2f}%\n")
                f.write(f"  - 平均有效率 (Effectiveness): {metrics['avg_effectiveness']:.2f}%\n")
                f.write(f"  - 平均准确率 (Precision): {metrics['avg_precision']:.2f}%\n")

            f.write("\n")

        # 计算跨数据集的总体统计
        f.write("\n" + "=" * 80 + "\n")
        f.write("## 跨数据集总体统计\n")
        f.write("=" * 80 + "\n\n")

        # 按算法汇总（加权平均）
        algo_aggregated = defaultdict(lambda: {
            'recall_weighted': 0, 'effectiveness_weighted': 0, 'precision_weighted': 0,
            'recall_simple': [], 'effectiveness_simple': [], 'precision_simple': [],
            'count': 0, 'dataset_count': 0
        })

        for dataset_name, algos in all_data.items():
            for algo_name, metrics in algos.items():
                weight = metrics['count']
                algo_aggregated[algo_name]['recall_weighted'] += metrics['avg_recall'] * weight
                algo_aggregated[algo_name]['effectiveness_weighted'] += metrics['avg_effectiveness'] * weight
                algo_aggregated[algo_name]['precision_weighted'] += metrics['avg_precision'] * weight
                algo_aggregated[algo_name]['recall_simple'].append(metrics['avg_recall'])
                algo_aggregated[algo_name]['effectiveness_simple'].append(metrics['avg_effectiveness'])
                algo_aggregated[algo_name]['precision_simple'].append(metrics['avg_precision'])
                algo_aggregated[algo_name]['count'] += weight
                algo_aggregated[algo_name]['dataset_count'] += 1

        # 输出总体统计
        for algo_name in sorted(algo_aggregated.keys()):
            data = algo_aggregated[algo_name]
            total_count = data['count']
            weighted_recall = data['recall_weighted'] / total_count if total_count > 0 else 0
            weighted_eff = data['effectiveness_weighted'] / total_count if total_count > 0 else 0
            weighted_prec = data['precision_weighted'] / total_count if total_count > 0 else 0

            f.write(f"\n### {algo_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  - 总评估图片对数: {total_count}\n")
            f.write(f"  - 加权平均召回率: {weighted_recall:.2f}%\n")
            f.write(f"  - 加权平均有效率: {weighted_eff:.2f}%\n")
            f.write(f"  - 加权平均准确率: {weighted_prec:.2f}%\n")
            f.write(f"  - 数据集覆盖: {data['dataset_count']} 个数据集\n")

        # 算法排名（基于加权平均）
        f.write("\n" + "=" * 80 + "\n")
        f.write("## 算法性能排名（加权平均）\n")
        f.write("=" * 80 + "\n\n")

        # 按召回率排名
        recall_ranking = sorted(algo_aggregated.items(),
                               key=lambda x: x[1]['recall_weighted']/x[1]['count'] if x[1]['count'] > 0 else 0,
                               reverse=True)

        f.write("### 按召回率排名 (Recall)\n")
        for rank, (algo_name, data) in enumerate(recall_ranking, 1):
            weighted_recall = data['recall_weighted']/data['count'] if data['count'] > 0 else 0
            f.write(f"{rank}. {algo_name}: {weighted_recall:.2f}%\n")

        # 按有效率排名
        effectiveness_ranking = sorted(algo_aggregated.items(),
                                      key=lambda x: x[1]['effectiveness_weighted']/x[1]['count'] if x[1]['count'] > 0 else 0,
                                      reverse=True)

        f.write("\n### 按有效率排名 (Effectiveness)\n")
        for rank, (algo_name, data) in enumerate(effectiveness_ranking, 1):
            weighted_eff = data['effectiveness_weighted']/data['count'] if data['count'] > 0 else 0
            f.write(f"{rank}. {algo_name}: {weighted_eff:.2f}%\n")

        # 按准确率排名
        precision_ranking = sorted(algo_aggregated.items(),
                                  key=lambda x: x[1]['precision_weighted']/x[1]['count'] if x[1]['count'] > 0 else 0,
                                  reverse=True)

        f.write("\n### 按准确率排名 (Precision)\n")
        for rank, (algo_name, data) in enumerate(precision_ranking, 1):
            weighted_prec = data['precision_weighted']/data['count'] if data['count'] > 0 else 0
            f.write(f"{rank}. {algo_name}: {weighted_prec:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 汇总报告已生成: {report_path}")
    print(f"\n发现 {len(all_data)} 个数据集，共 {len(algo_aggregated)} 种算法配置")

    # 打印简要摘要
    print("\n简要摘要（加权平均）:")
    for algo_name in sorted(algo_aggregated.keys()):
        data = algo_aggregated[algo_name]
        weighted_recall = data['recall_weighted']/data['count'] if data['count'] > 0 else 0
        print(f"  {algo_name}: 召回率 {weighted_recall:.2f}%")

if __name__ == "__main__":
    main()
