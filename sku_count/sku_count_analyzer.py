#!/usr/bin/env python3
"""
SKU计数分析器 - 统计去重前后的物体总数

分析检测结果和匹配摘要，计算：
1. 去重前总物体数（所有图像检测结果的总和）
2. 重复匹配的物体数
3. 去重后的总物体数
"""

import json
import re
from pathlib import Path
from typing import Dict
from collections import defaultdict
from datetime import datetime

class SKUCountAnalyzer:
    """SKU计数分析器"""
    
    def __init__(self, detection_dir: str, summary_dir: str = "output_pt"):
        """
        初始化分析器
        
        Args:
            detection_dir: 检测结果目录路径
            summary_dir: 匹配摘要文件目录路径（包含多个参考索引子目录）
        """
        self.detection_dir = Path(detection_dir)
        self.summary_dir = Path(summary_dir)
        
    def count_total_detections(self) -> Dict[str, int]:
        """统计所有检测结果中的物体总数"""
        total_count = 0
        per_image_count = {}
        
        print("=== 统计检测结果 ===")
        detection_files = sorted(self.detection_dir.glob("*.json"), 
                               key=lambda x: int(x.stem) if x.stem.isdigit() else 999)
        
        for detection_file in detection_files:
            image_num = detection_file.stem
            
            try:
                with open(detection_file, 'r', encoding='utf-8') as f:
                    detection_data = json.load(f)
                
                if isinstance(detection_data, list) and len(detection_data) > 0:
                    objects = detection_data[0].get('objects', [])
                    object_count = len(objects)
                    
                    per_image_count[image_num] = object_count
                    total_count += object_count
                    
                    print(f"图像 {image_num}: {object_count} 个物体")
                    
            except Exception as e:
                print(f"读取文件 {detection_file} 出错: {e}")
                per_image_count[image_num] = 0
        
        print(f"\n去重前总物体数: {total_count}")
        return {
            'total': total_count,
            'per_image': per_image_count,
            'num_images': len(per_image_count)
        }
    
    def analyze_all_matches(self) -> Dict:
        """分析所有参考索引的匹配摘要文件"""
        print("\n=== 分析所有匹配结果 ===")
        
        all_matched_pairs = []
        all_unique_refs = set()
        all_target_images = set()
        summary_files_found = []
        
        # 查找所有summary文件
        for ref_idx in range(13):  # 0-12
            summary_file = self.summary_dir / str(ref_idx) / "matching_summary.txt"
            if summary_file.exists():
                summary_files_found.append((ref_idx, summary_file))
        
        print(f"找到 {len(summary_files_found)} 个匹配摘要文件")
        
        # 分析每个summary文件
        for ref_idx, summary_file in summary_files_found:
            print(f"\n分析参考索引 {ref_idx} 的匹配结果...")
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取匹配记录
            match_pattern = r'Matched ref (\d+) → target (\d+) \(hit ratio: ([\d.]+) (\d+)/(\d+)\)'
            matches = re.findall(match_pattern, content)
            
            # 提取目标图像信息
            target_image_pattern = r'target image (\d+)'
            
            ref_matches_count = len(matches)
            print(f"  参考索引 {ref_idx}: {ref_matches_count} 个匹配")
            
            # 为每个匹配记录关联对应的目标图像
            current_target_img = None
            match_idx = 0
            
            for line in content.split('\n'):
                target_match = re.search(target_image_pattern, line)
                if target_match:
                    current_target_img = int(target_match.group(1))
                elif 'Matched ref' in line and match_idx < len(matches):
                    match = matches[match_idx]
                    ref_id, target_id, hit_ratio, matched_points, total_points = match
                    ref_id = int(ref_id)
                    target_id = int(target_id)
                    hit_ratio = float(hit_ratio)
                    matched_points = int(matched_points)
                    total_points = int(total_points)
                    
                    # 添加参考索引信息
                    match_info = {
                        'ref_idx': ref_idx,
                        'ref_id': ref_id,
                        'target_idx': current_target_img,
                        'target_id': target_id,
                        'hit_ratio': hit_ratio,
                        'matched_points': matched_points,
                        'total_points': total_points
                    }
                    
                    all_matched_pairs.append(match_info)
                    all_unique_refs.add((ref_idx, ref_id))
                    match_idx += 1
            
        
        print(f"\n汇总统计:")
        print(f"  总匹配对数: {len(all_matched_pairs)}")
        print(f"  参与匹配的唯一参考物体数: {len(all_unique_refs)}")
        print(f"  涉及的目标图像数: {len(all_target_images)}")
        
        return {
            'matched_pairs': len(all_matched_pairs),
            'unique_refs': len(all_unique_refs),
            'target_images': len(all_target_images),
            'pairs': all_matched_pairs,
            'summary_files_count': len(summary_files_found)
        }
    
    def calculate_duplicates(self, detection_stats: Dict, match_stats: Dict) -> Dict:
        """计算重复物体数量 - 修正版"""
        print("\n=== 计算重复统计（修正版）===")
        
        print(f"处理了 {match_stats['summary_files_count']} 个参考索引的匹配结果")
        print(f"总匹配对数: {match_stats['matched_pairs']}")
        print(f"参与匹配的唯一参考物体数: {match_stats['unique_refs']}")
        
        # 重新分析匹配关系：构建连通组件
        # 每个连通组件代表同一个真实物体在不同图像中的所有检测结果
        
        # 1. 构建图的邻接关系
        adjacency = defaultdict(set)
        all_objects = set()  # 存储所有(image_idx, object_id)对
        
        # 从检测结果中添加所有实际检测到的物体
        for img_num, count in detection_stats['per_image'].items():
            image_num = int(img_num)
            for obj_idx in range(count):
                all_objects.add((image_num, obj_idx))
        
        print(f"从检测结果中加载了 {len(all_objects)} 个物体")
        
        # 添加匹配关系作为边
        for match in match_stats['pairs']:
            ref_img = match['ref_idx']
            ref_obj = match['ref_id']
            target_img = match.get('target_idx')
            target_obj = match['target_id']
            
            if target_img is not None:
                obj1 = (ref_img, ref_obj)
                obj2 = (target_img, target_obj)
                adjacency[obj1].add(obj2)
                adjacency[obj2].add(obj1)
        
        # 2. 使用DFS找连通组件
        visited = set()
        connected_components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            for neighbor in adjacency[node]:
                dfs(neighbor, component)
        
        for obj in all_objects:
            if obj not in visited:
                component = []
                dfs(obj, component)
                connected_components.append(component)
        
        # 3. 计算统计数据
        total_before = len(all_objects)
        unique_objects = len(connected_components)  # 连通组件数=去重后的真实物体数
        duplicate_count = total_before - unique_objects
        
        # 统计连通组件大小分布
        component_sizes = [len(comp) for comp in connected_components]
        single_object_components = sum(1 for size in component_sizes if size == 1)
        multi_object_components = sum(1 for size in component_sizes if size > 1)
        max_component_size = max(component_sizes) if component_sizes else 0
        
        print(f"\n连通组件分析:")
        print(f"  总连通组件数（唯一物体数）: {unique_objects}")
        print(f"  单独物体组件数: {single_object_components}")
        print(f"  多检测物体组件数: {multi_object_components}")
        print(f"  最大组件大小: {max_component_size}")
        
        return {
            'total_before_dedup': total_before,
            'duplicate_count': duplicate_count,
            'total_after_dedup': unique_objects,
            'summary_files_processed': match_stats['summary_files_count'],
            'unique_ref_objects': match_stats['unique_refs'],
            'connected_components': len(connected_components),
            'single_object_components': single_object_components,
            'multi_object_components': multi_object_components,
            'max_component_size': max_component_size
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """生成完整的计数分析报告"""
        print("\n" + "="*60)
        print("SKU计数分析报告")
        print("="*60)
        
        # 统计检测结果
        detection_stats = self.count_total_detections()
        
        # 分析匹配结果
        match_stats = self.analyze_all_matches()
        
        # 计算重复统计
        duplicate_stats = self.calculate_duplicates(detection_stats, match_stats)
        
        # 生成报告内容
        report_lines = []
        report_lines.append("SKU计数分析报告")
        report_lines.append("="*60)
        report_lines.append("")
        
        report_lines.append("1. 检测结果统计:")
        report_lines.append(f"   - 处理图像数: {detection_stats['num_images']}")
        report_lines.append(f"   - 去重前总物体数: {detection_stats['total']}")
        
        # 按图像显示详情
        report_lines.append("\n   各图像物体数详情:")
        for img_num in sorted(detection_stats['per_image'].keys(), key=lambda x: int(x) if x.isdigit() else 999):
            count = detection_stats['per_image'][img_num]
            report_lines.append(f"   图像 {img_num}: {count} 个物体")
        
        report_lines.append("")
        report_lines.append("2. 匹配结果统计:")
        report_lines.append(f"   - 处理的参考索引数: {duplicate_stats['summary_files_processed']}")
        report_lines.append(f"   - 参与匹配的唯一参考物体数: {duplicate_stats['unique_ref_objects']}")
        report_lines.append(f"   - 匹配到的目标图像数: {match_stats['target_images']}")
        report_lines.append(f"   - 总匹配对数: {match_stats['matched_pairs']}")
        
        report_lines.append("")
        report_lines.append("3. 去重计算（基于连通组件）:")
        report_lines.append(f"   - 去重前总物体数: {duplicate_stats['total_before_dedup']}")
        report_lines.append(f"   - 去重后唯一物体数: {duplicate_stats['total_after_dedup']}")
        report_lines.append(f"   - 重复检测数: {duplicate_stats['duplicate_count']}")
        
        dedup_percentage = (duplicate_stats['duplicate_count'] / duplicate_stats['total_before_dedup']) * 100
        report_lines.append(f"   - 去重率: {dedup_percentage:.1f}%")
        
        report_lines.append("")
        report_lines.append("4. 连通组件分析:")
        report_lines.append(f"   - 总连通组件数: {duplicate_stats['connected_components']}")
        report_lines.append(f"   - 独立物体数: {duplicate_stats['single_object_components']}")
        report_lines.append(f"   - 多检测物体数: {duplicate_stats['multi_object_components']}")
        report_lines.append(f"   - 最大组件大小: {duplicate_stats['max_component_size']}")
        
        report_lines.append("")
        report_lines.append("5. 计算说明:")
        report_lines.append("   - 使用图论连通组件算法进行去重")
        report_lines.append("   - 每个连通组件代表一个真实的物体")
        report_lines.append("   - 匹配关系作为图的边连接同一物体的不同检测结果")
        report_lines.append("   - 去重后数量 = 连通组件数量")
        
        report_content = "\n".join(report_lines)
        
        # 打印到控制台
        print("\n" + report_content)
        
        # 保存到文件
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sku_count/sku_count_analysis_{timestamp}.txt"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n报告已保存到: {output_path.absolute()}")
        
        return report_content

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SKU计数分析器")
    parser.add_argument("--detection_dir", type=str, 
                       default="../imdata/detections_results",
                       help="检测结果目录路径")
    parser.add_argument("--summary_dir", type=str,
                       default="output_pt", 
                       help="匹配摘要目录路径（包含多个参考索引子目录）")
    parser.add_argument("--output_file", type=str,
                       help="输出报告文件路径（可选）")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SKUCountAnalyzer(args.detection_dir, args.summary_dir)
    
    # 生成报告
    analyzer.generate_report(args.output_file)

if __name__ == "__main__":
    main()