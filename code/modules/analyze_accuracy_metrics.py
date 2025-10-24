#!/usr/bin/env python3
"""
批量分析imdata0911目录下floor_display2-12的准确性评估指标
计算所有txt文件中总体性能指标的加权平均值
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def extract_metrics_from_file(file_path: str) -> Optional[Dict[str, float]]:
    """
    从txt文件中提取性能指标
    
    返回格式:
    {
        'recall': float,     # 召回率百分比
        'recall_num': int,   # 召回率分子
        'recall_den': int,   # 召回率分母
        'effectiveness': float,  # VGGT有效率百分比
        'effectiveness_num': int,  # 有效率分子
        'effectiveness_den': int,  # 有效率分母
        'precision': float,      # 精确率百分比
        'precision_num': int,    # 精确率分子
        'precision_den': int     # 精确率分母
    }
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics = {}
        
        # 提取总体召回率 (Recall): 80.00% (28/35)
        recall_pattern = r'总体召回率.*?(\d+\.?\d*)%.*?\((\d+)/(\d+)\)'
        recall_match = re.search(recall_pattern, content)
        if recall_match:
            metrics['recall'] = float(recall_match.group(1))
            metrics['recall_num'] = int(recall_match.group(2))
            metrics['recall_den'] = int(recall_match.group(3))
        
        # 提取VGGT有效率 (Effectiveness): 52.83% (28/53)
        effectiveness_pattern = r'VGGT有效率.*?(\d+\.?\d*)%.*?\((\d+)/(\d+)\)'
        effectiveness_match = re.search(effectiveness_pattern, content)
        if effectiveness_match:
            metrics['effectiveness'] = float(effectiveness_match.group(1))
            metrics['effectiveness_num'] = int(effectiveness_match.group(2))
            metrics['effectiveness_den'] = int(effectiveness_match.group(3))
        
        # 提取Reference ID映射准确率 (Precision): 87.10% (27/31)
        precision_pattern = r'Reference ID映射准确率.*?(\d+\.?\d*)%.*?\((\d+)/(\d+)\)'
        precision_match = re.search(precision_pattern, content)
        if precision_match:
            metrics['precision'] = float(precision_match.group(1))
            metrics['precision_num'] = int(precision_match.group(2))
            metrics['precision_den'] = int(precision_match.group(3))
        
        # 验证是否提取到所有指标
        required_keys = ['recall', 'effectiveness', 'precision']
        if all(key in metrics for key in required_keys):
            return metrics
        else:
            print(f"WARNING: {file_path} 中未找到完整的性能指标")
            return None
            
    except (FileNotFoundError, UnicodeDecodeError, PermissionError) as e:
        print(f"ERROR: 读取文件 {file_path} 失败: {e}")
        return None

def find_all_accuracy_files(base_dir: str) -> List[str]:
    """
    查找所有floor_display2-12目录下accuracy_evaluation中的txt文件
    """
    base_path = Path(base_dir)
    txt_files = []
    
    for i in range(2, 13):  # floor_display2 到 floor_display12
        floor_dir = base_path / f"floor_display{i}" / "accuracy_evaluation"
        if floor_dir.exists():
            for txt_file in floor_dir.glob("*.txt"):
                if txt_file.name != "summary.txt":  # 跳过汇总文件
                    txt_files.append(str(txt_file))
    
    return sorted(txt_files)

def calculate_weighted_averages(all_metrics: List[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
    """
    计算加权平均值
    权重基于每个指标的分母（样本数量）
    过滤掉总体准确率小于50%的文件
    """
    # 过滤低准确率文件
    filtered_metrics = []
    excluded_files = []
    
    for file_path, metrics in all_metrics:
        if not metrics:
            continue
        
        # 计算总体准确率 (这里使用召回率作为总体准确率的代表)
        overall_accuracy = metrics.get('recall', 0)
        
        if overall_accuracy >= 50.0:  # 总体准确率大于等于50%
            filtered_metrics.append((file_path, metrics))
        else:
            excluded_files.append((file_path, overall_accuracy))
    
    print(f"\n过滤结果:")
    print(f"  - 保留文件: {len(filtered_metrics)} 个")
    print(f"  - 排除文件: {len(excluded_files)} 个 (总体准确率 < 50%)")
    
    if excluded_files:
        print(f"\n排除的低准确率文件:")
        for file_path, accuracy in excluded_files:
            print(f"  - {file_path}: {accuracy:.2f}%")
    
    # 分别统计三个指标的加权总和和总权重
    recall_weighted_sum = 0
    recall_total_weight = 0
    
    effectiveness_weighted_sum = 0
    effectiveness_total_weight = 0
    
    precision_weighted_sum = 0
    precision_total_weight = 0
    
    # 分别统计原始数据的总和，用于重新计算百分比
    recall_total_num = 0
    recall_total_den = 0
    
    effectiveness_total_num = 0
    effectiveness_total_den = 0
    
    precision_total_num = 0
    precision_total_den = 0
    
    for file_path, metrics in filtered_metrics:
        if not metrics:
            continue
        
        # 召回率加权计算
        if all(key in metrics for key in ['recall', 'recall_den', 'recall_num']):
            weight = metrics['recall_den']  # 使用分母作为权重
            recall_weighted_sum += metrics['recall'] * weight
            recall_total_weight += weight
            
            recall_total_num += metrics['recall_num']
            recall_total_den += metrics['recall_den']
        
        # 有效率加权计算
        if all(key in metrics for key in ['effectiveness', 'effectiveness_den', 'effectiveness_num']):
            weight = metrics['effectiveness_den']
            effectiveness_weighted_sum += metrics['effectiveness'] * weight
            effectiveness_total_weight += weight
            
            effectiveness_total_num += metrics['effectiveness_num']
            effectiveness_total_den += metrics['effectiveness_den']
        
        # 精确率加权计算
        if all(key in metrics for key in ['precision', 'precision_den', 'precision_num']):
            weight = metrics['precision_den']
            precision_weighted_sum += metrics['precision'] * weight
            precision_total_weight += weight
            
            precision_total_num += metrics['precision_num']
            precision_total_den += metrics['precision_den']
    
    # 计算加权平均值
    result = {}
    
    if recall_total_weight > 0:
        result['recall_weighted_avg'] = recall_weighted_sum / recall_total_weight
        result['recall_overall'] = (recall_total_num / recall_total_den * 100) if recall_total_den > 0 else 0
        result['recall_total_correct'] = recall_total_num
        result['recall_total_samples'] = recall_total_den
    
    if effectiveness_total_weight > 0:
        result['effectiveness_weighted_avg'] = effectiveness_weighted_sum / effectiveness_total_weight
        result['effectiveness_overall'] = (effectiveness_total_num / effectiveness_total_den * 100) if effectiveness_total_den > 0 else 0
        result['effectiveness_total_correct'] = effectiveness_total_num
        result['effectiveness_total_samples'] = effectiveness_total_den
    
    if precision_total_weight > 0:
        result['precision_weighted_avg'] = precision_weighted_sum / precision_total_weight
        result['precision_overall'] = (precision_total_num / precision_total_den * 100) if precision_total_den > 0 else 0
        result['precision_total_correct'] = precision_total_num
        result['precision_total_samples'] = precision_total_den
    
    return result

def main():
    """主函数"""
    # 默认使用仓库根目录下的 imdata0911，可通过 --base-dir 覆盖
    import argparse
    default_base = str(Path(__file__).resolve().parents[1] / 'imdata0911')
    parser = argparse.ArgumentParser(description="批量汇总 accuracy_evaluation 指标")
    parser.add_argument("--base-dir", type=str, default=default_base,
                        help="包含 floor_display*/accuracy_evaluation 的根目录（默认：<repo>/imdata0911）")
    args = parser.parse_args()
    base_dir = args.base_dir
    
    print("扫描floor_display2-12的accuracy_evaluation目录...")
    
    # 查找所有txt文件
    all_txt_files = find_all_accuracy_files(base_dir)
    print(f"找到 {len(all_txt_files)} 个评估文件")
    
    # 提取所有文件的指标
    all_metrics = []
    successful_extractions = 0
    
    for txt_file in all_txt_files:
        relative_path = os.path.relpath(txt_file, base_dir)
        print(f"处理: {relative_path}")
        
        metrics = extract_metrics_from_file(txt_file)
        if metrics:
            all_metrics.append((relative_path, metrics))
            successful_extractions += 1
            print(f"   召回率: {metrics['recall']:.2f}%, 有效率: {metrics['effectiveness']:.2f}%, 精确率: {metrics['precision']:.2f}%")
        else:
            all_metrics.append((relative_path, None))
            print(f"   提取失败")
    
    print(f"\n成功提取 {successful_extractions}/{len(all_txt_files)} 个文件的指标")
    
    if successful_extractions == 0:
        print("ERROR: 没有成功提取任何指标")
        return
    
    # 计算加权平均值
    print("\n计算加权平均值...")
    weighted_averages = calculate_weighted_averages(all_metrics)
    
    # 输出结果
    print("\n" + "="*60)
    print("Floor Display 2-12 整体性能指标汇总")
    print("="*60)
    
    if 'recall_weighted_avg' in weighted_averages:
        print(f"总体召回率 (Recall):")
        print(f"  - 加权平均: {weighted_averages['recall_weighted_avg']:.2f}%")
        print(f"  - 整体计算: {weighted_averages['recall_overall']:.2f}% ({weighted_averages['recall_total_correct']}/{weighted_averages['recall_total_samples']})")
    
    if 'effectiveness_weighted_avg' in weighted_averages:
        print(f"VGGT有效率 (Effectiveness):")
        print(f"  - 加权平均: {weighted_averages['effectiveness_weighted_avg']:.2f}%")
        print(f"  - 整体计算: {weighted_averages['effectiveness_overall']:.2f}% ({weighted_averages['effectiveness_total_correct']}/{weighted_averages['effectiveness_total_samples']})")
    
    if 'precision_weighted_avg' in weighted_averages:
        print(f"Reference ID映射准确率 (Precision):")
        print(f"  - 加权平均: {weighted_averages['precision_weighted_avg']:.2f}%")
        print(f"  - 整体计算: {weighted_averages['precision_overall']:.2f}% ({weighted_averages['precision_total_correct']}/{weighted_averages['precision_total_samples']})")
    
    print("="*60)
    
    # 统计各个floor_display的文件数量
    print(f"\n按目录分组统计:")
    floor_counts = {}
    for relative_path, _ in all_metrics:
        floor_name = relative_path.split('/')[0]
        floor_counts[floor_name] = floor_counts.get(floor_name, 0) + 1
    
    for floor_name in sorted(floor_counts.keys()):
        print(f"  - {floor_name}: {floor_counts[floor_name]} 个文件")
    
    print(f"\n处理完成！共分析了 {len(all_txt_files)} 个评估文件")

if __name__ == "__main__":
    main()
