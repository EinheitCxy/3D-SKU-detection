"""
SKU匹配准确性标注系统

该系统用于比较人工标注的SKU对应关系与VGGT系统输出的匹配结果，
计算准确性指标，包括精确度(Precision)、召回率(Recall)和F1分数。
"""

import pandas as pd
import re
import os
import csv
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import argparse
from datetime import datetime

 
class AccuracyAnnotator:
    """准确性标注器类"""
    
    def __init__(self):
        self.ground_truth = {}  # 人工标注的真实数据
        self.vggt_results = {}  # VGGT系统结果
        self.metrics = {}       # 评估指标
    
    def load_ground_truth(self, csv_path: str, dataset_filter: str = None) -> None:
        try:
            df = pd.read_csv(csv_path)
            
            # 如果指定了数据集过滤器，则过滤数据
            if dataset_filter:
                df = df[df['img_src'].str.contains(dataset_filter)]
            
            # 解析数据并组织为字典格式
            for _, row in df.iterrows():
                ref_img = row['reference_img']
                target_img = row['target_img']
                image_pair = f"{ref_img}_to_{target_img}"
                
                if image_pair not in self.ground_truth:
                    self.ground_truth[image_pair] = []
                
                self.ground_truth[image_pair].append({
                    'reference_id': int(row['reference_id']),
                    'target_id': int(row['target_id']),
                    'source_data': row['img_src']
                })

        except (FileNotFoundError, PermissionError, csv.Error, UnicodeDecodeError) as e:
            print(f"加载人工标注数据失败: {e}")
            raise
    
    def load_vggt_results(self, result_path: str) -> None:        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 从文件路径推断参考图片编号
            import os
            parent_dir = os.path.basename(os.path.dirname(result_path))
            ref_base_index = int(parent_dir)
            actual_ref = ref_base_index + 1
            
            # 基于人工标注中的图片对，查找对应的VGGT匹配结果
            expected_pairs = []
            for pair_name in self.ground_truth.keys():
                ref_img_str, target_img_str = pair_name.split('_to_')
                if int(ref_img_str) == actual_ref:
                    expected_pairs.append((pair_name, int(target_img_str)))
            
            if not expected_pairs:
                return
            
            lines = content.split('\n')
            found_pattern = r'Found (\d+) matches in image (\d+)'
            match_pattern = r'Matched ref (\d+) → target (\d+) \(hit ratio: ([\d.]+) (\d+)/(\d+)\)'
            
            # 找到所有"Found X matches in image Y"行的位置
            found_lines = []
            for i, line in enumerate(lines):
                found_match = re.search(found_pattern, line)
                if found_match:
                    match_count, target_img = found_match.groups()
                    found_lines.append({
                        'line_idx': i,
                        'target_img': int(target_img),
                        'actual_target': int(target_img) + 1
                    })
            
            # 为每个found标记提取其上方的匹配内容
            sections = []
            for i, found_info in enumerate(found_lines):
                actual_target = found_info['actual_target']
                pair_name = f"{actual_ref}_to_{actual_target}"
                
                if i == 0:
                    section_start = 0
                else:
                    section_start = found_lines[i-1]['line_idx'] + 1
                
                section_end = found_info['line_idx']
                
                # 从section中提取匹配
                matches = []
                for line_idx in range(section_start, section_end):
                    if line_idx < len(lines):
                        line = lines[line_idx].strip()
                        if line:
                            match_result = re.search(match_pattern, line)
                            if match_result:
                                ref_id, target_id, hit_ratio, matched_points, total_points = match_result.groups()
                                matches.append({
                                    'reference_id': int(ref_id),
                                    'target_id': int(target_id),
                                    'hit_ratio': float(hit_ratio),
                                    'matched_points': int(matched_points),
                                    'total_points': int(total_points)
                                })
                
                sections.append({
                    'actual_target': actual_target,
                    'pair_name': pair_name,
                    'matches': matches
                })
            
            # 为每个期望的图片对查找匹配结果
            for pair_name, expected_target in expected_pairs:
                target_section = None
                for section in sections:
                    if section['actual_target'] == expected_target:
                        target_section = section
                        break

                if target_section and target_section['matches']:
                    self.vggt_results[pair_name] = target_section['matches']

        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            print(f"加载VGGT结果失败: {e}")
            raise
    
    def extract_image_pair_from_vggt_path(self, result_path: str) -> str:
        """
        从VGGT结果文件路径推断图片对信息
        
        Args:
            result_path: VGGT结果文件路径
            
        Returns:
            图片对标识符，如"0_to_1"
        """
        # 尝试从路径中提取图片对信息
        # 这里可以根据实际的文件组织结构进行调整
        if "output_pt/0" in result_path:
            return "0_to_1"  # 默认假设是参考图像0到目标图像1
        return "0_to_1"
    
    def calculate_metrics(self, image_pair: str) -> Dict:
        """
        计算指定图片对的准确性指标
        
        Args:
            image_pair: 图片对标识符
            
        Returns:
            包含各项指标的字典
        """
        if image_pair not in self.ground_truth:
            print(f"警告: 未找到图片对 {image_pair} 的人工标注数据")
            return {}
        
        if image_pair not in self.vggt_results:
            print(f"警告: 未找到图片对 {image_pair} 的VGGT结果")
            return {}
        
        # 获取人工标注和VGGT结果
        gt_matches = self.ground_truth[image_pair]
        vggt_matches = self.vggt_results[image_pair]
        
        # 转换为集合格式进行比较
        gt_set = {(match['reference_id'], match['target_id']) for match in gt_matches}
        vggt_set = {(match['reference_id'], match['target_id']) for match in vggt_matches}
        
        # 转换为字典格式进行reference ID映射分析
        gt_mapping = {match['reference_id']: match['target_id'] for match in gt_matches}
        vggt_mapping = {match['reference_id']: match['target_id'] for match in vggt_matches}
        
        # 重新定义评估指标（人工标注不存在的不算错）
        true_positives = len(gt_set & vggt_set)  # 人工标注存在且VGGT也预测的匹配
        false_negatives = len(gt_set - vggt_set)  # 人工标注存在但VGGT未预测的匹配（遗漏）
        # 注意：不计算false_positives，因为人工标注可能不完整
        
        # 计算基于reference ID的映射准确率
        # 找到VGGT和人工标注都有映射的reference ID
        common_ref_ids = set(gt_mapping.keys()) & set(vggt_mapping.keys())
        ref_mapping_correct = 0
        ref_mapping_total = len(common_ref_ids)
        
        # 分类不同类型的匹配错误
        wrong_mappings = []  # 错误映射：同一ref_id映射到不同target_id
        
        for ref_id in common_ref_ids:
            if gt_mapping[ref_id] == vggt_mapping[ref_id]:
                ref_mapping_correct += 1
            else:
                # 记录错误映射
                wrong_mappings.append((ref_id, vggt_mapping[ref_id], gt_mapping[ref_id]))
        
        # 重新计算额外预测和遗漏，排除错误映射的情况
        vggt_extra_matches_filtered = []
        missed_matches_filtered = []
        
        for match in (vggt_set - gt_set):
            ref_id, target_id = match
            # 如果这个ref_id在人工标注中也存在但映射不同，则归类为错误映射，不算额外预测
            if ref_id in gt_mapping and gt_mapping[ref_id] != target_id:
                continue  # 已经在wrong_mappings中记录
            else:
                vggt_extra_matches_filtered.append(match)
        
        for match in (gt_set - vggt_set):
            ref_id, target_id = match
            # 如果这个ref_id在VGGT中也存在但映射不同，则归类为错误映射，不算遗漏
            if ref_id in vggt_mapping and vggt_mapping[ref_id] != target_id:
                continue  # 已经在wrong_mappings中记录
            else:
                missed_matches_filtered.append(match)
        
        # 计算召回率（基于人工标注的完整性）
        recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0
        
        # 计算VGGT预测中的有效比例（在人工标注范围内的准确性）
        vggt_effectiveness = true_positives / len(vggt_set) if len(vggt_set) > 0 else 0
        
        # 计算基于reference ID映射的准确率
        ref_mapping_precision = ref_mapping_correct / ref_mapping_total if ref_mapping_total > 0 else 0
        
        metrics = {
            'image_pair': image_pair,
            'ground_truth_matches': len(gt_matches),
            'vggt_matches': len(vggt_matches),
            'true_positives': true_positives,
            'false_negatives': false_negatives,
            'recall': recall,
            'vggt_effectiveness': vggt_effectiveness,
            'ref_mapping_precision': ref_mapping_precision,
            'ref_mapping_correct': ref_mapping_correct,
            'ref_mapping_total': ref_mapping_total,
            'correct_matches': list(gt_set & vggt_set),
            'missed_matches': missed_matches_filtered,  # 使用过滤后的遗漏匹配
            'vggt_extra_matches': vggt_extra_matches_filtered,  # 使用过滤后的额外预测
            'wrong_mappings': wrong_mappings  # 新增：错误映射
        }
        
        return metrics
    
    def generate_report(self, output_path: str = None) -> str:
        """
        生成详细的评估报告
        
        Args:
            output_path: 输出文件路径，如果为None则只返回报告内容
            
        Returns:
            报告内容字符串
        """
        report_lines = [
            "=" * 80,
            "SKU匹配准确性评估报告",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # 计算所有图片对的指标
        all_metrics = []
        for image_pair in self.ground_truth.keys():
            if image_pair in self.vggt_results:
                metrics = self.calculate_metrics(image_pair)
                if metrics:
                    all_metrics.append(metrics)
        
        if not all_metrics:
            report_lines.extend([
                "错误: 未找到可比较的图片对数据",
                "请检查人工标注数据和VGGT结果是否匹配"
            ])
        else:
            # 计算总体指标
            total_gt = sum(m['ground_truth_matches'] for m in all_metrics)
            total_vggt = sum(m['vggt_matches'] for m in all_metrics)
            total_tp = sum(m['true_positives'] for m in all_metrics)
            total_fn = sum(m['false_negatives'] for m in all_metrics)
            total_extra = sum(len(m['vggt_extra_matches']) for m in all_metrics)
            total_wrong_mappings = sum(len(m['wrong_mappings']) for m in all_metrics)
            
            # 计算总体reference ID映射指标
            total_ref_mapping_correct = sum(m['ref_mapping_correct'] for m in all_metrics)
            total_ref_mapping_total = sum(m['ref_mapping_total'] for m in all_metrics)
            
            overall_recall = total_tp / total_gt if total_gt > 0 else 0
            overall_effectiveness = total_tp / total_vggt if total_vggt > 0 else 0
            overall_ref_mapping_precision = total_ref_mapping_correct / total_ref_mapping_total if total_ref_mapping_total > 0 else 0
            
            report_lines.extend([
                "总体性能指标:",
                f"  总体召回率 (Recall): {overall_recall:.2%} ({total_tp}/{total_gt})",
                f"  VGGT有效率 (Effectiveness): {overall_effectiveness:.2%} ({total_tp}/{total_vggt})",
                f"  Reference ID映射准确率 (Precision): {overall_ref_mapping_precision:.2%} ({total_ref_mapping_correct}/{total_ref_mapping_total})",
                "",
                "详细统计:",
                f"  人工标注总匹配数: {total_gt}",
                f"  VGGT预测总匹配数: {total_vggt}",
                f"  正确匹配数: {total_tp}",
                f"  遗漏匹配数: {total_fn}",
                f"  VGGT额外预测数: {total_extra}",
                f"  错误映射数: {total_wrong_mappings}",
                f"  共同Reference ID数量: {total_ref_mapping_total}",
                f"  Reference ID映射正确数: {total_ref_mapping_correct}",
                "",
                "说明: VGGT额外预测的匹配不被视为错误，因为人工标注可能不完整",
                "Reference ID映射准确率: 当VGGT和人工标注都对同一ref ID有映射时，映射目标是否一致",
                ""
            ])
            
            # 添加详细匹配分析
            for metrics in all_metrics:
                if metrics['image_pair']:  # 确保有图片对数据
                    report_lines.extend([
                        f"图片对 {metrics['image_pair']} 详细分析:",
                        "-" * 50
                    ])
                    
                    # VGGT额外预测的匹配
                    if metrics['vggt_extra_matches']:
                        report_lines.extend([
                            "VGGT额外检出对 (人工标注中不存在):",
                            f"  {metrics['vggt_extra_matches']}",
                            ""
                        ])
                    
                    # 遗漏的匹配
                    if metrics['missed_matches']:
                        report_lines.extend([
                            "遗漏检出对 (人工标注存在但VGGT未检出):",
                            f"  {metrics['missed_matches']}",
                            ""
                        ])
                    
                    # 错误映射
                    if metrics['wrong_mappings']:
                        report_lines.extend([
                            "错误检出对 (同一ref_id映射到不同target_id):",
                            "  格式: (ref_id, VGGT预测的target_id, 人工标注的target_id)"
                        ])
                        for ref_id, vggt_target, gt_target in metrics['wrong_mappings']:
                            report_lines.append(f"  ref_{ref_id}: VGGT预测→{vggt_target}, 人工标注→{gt_target}")
                        report_lines.append("")
                    
                    # 正确匹配
                    if metrics['correct_matches']:
                        report_lines.extend([
                            "正确检出对 (人工标注和VGGT都检出):",
                            f"  {metrics['correct_matches']}",
                            ""
                        ])
                    
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # 如果指定了输出路径，则写入文件
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
            except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                print(f"保存报告失败: {e}")

        return report_content


def main():
    parser = argparse.ArgumentParser(description='SKU匹配准确性标注系统')
    parser.add_argument('--benchmark-csv', required=True,
                       help='人工标注数据集CSV文件路径')
    parser.add_argument('--vggt-result', required=True,
                       help='VGGT系统输出的匹配结果文件路径')
    parser.add_argument('--dataset-filter', default=None,
                       help='数据集过滤器')
    parser.add_argument('--output', default=None,
                       help='输出报告文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.benchmark_csv):
        print(f"错误: 人工标注文件不存在: {args.benchmark_csv}")
        return
    
    if not os.path.exists(args.vggt_result):
        print(f"错误: VGGT结果文件不存在: {args.vggt_result}")
        return
    
    # 创建标注器实例
    annotator = AccuracyAnnotator()
    
    try:
        # 加载数据
        annotator.load_ground_truth(args.benchmark_csv, args.dataset_filter)
        annotator.load_vggt_results(args.vggt_result)
        
        # 生成报告
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            report = annotator.generate_report(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_output = f"accuracy_report_{timestamp}.txt"
            report = annotator.generate_report(default_output)

    except (ValueError, KeyError, IndexError) as e:
        print(f"评估过程出错: {e}")
        return


if __name__ == "__main__":
    main()