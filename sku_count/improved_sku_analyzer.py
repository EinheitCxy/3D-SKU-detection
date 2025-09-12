#!/usr/bin/env python3
"""
改进的SKU计数分析器 - 处理一对多匹配问题
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

class ImprovedSKUCountAnalyzer:
    """改进的SKU计数分析器 - 解决过度聚类问题"""
    
    def __init__(self, detection_dir: str, summary_dir: str = "output_pt"):
        self.detection_dir = Path(detection_dir)
        self.summary_dir = Path(summary_dir)
        
    def filter_best_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        过滤匹配结果，每个物体只保留最佳匹配
        
        策略：
        1. 每个ref只保留hit_ratio最高的target
        2. 每个target只能被最高hit_ratio的ref匹配
        """
        print("\n=== 过滤最佳匹配 ===")
        
        # 按ref_obj分组
        ref_groups = defaultdict(list)
        for match in matches:
            key = (match['ref_idx'], match['ref_id'])
            ref_groups[key].append(match)
        
        # 为每个ref选择最佳target
        ref_best_matches = []
        for ref_key, ref_matches in ref_groups.items():
            best_match = max(ref_matches, key=lambda x: x['hit_ratio'])
            ref_best_matches.append(best_match)
            print(f"Ref {ref_key}: 选择最佳匹配 target({best_match['target_idx']},{best_match['target_id']}) hit_ratio={best_match['hit_ratio']:.3f}")
        
        # 按target分组，解决多个ref争夺同一target的冲突
        target_groups = defaultdict(list)
        for match in ref_best_matches:
            key = (match['target_idx'], match['target_id'])
            target_groups[key].append(match)
        
        # 为每个target选择最佳ref
        final_matches = []
        for target_key, target_matches in target_groups.items():
            if len(target_matches) > 1:
                best_match = max(target_matches, key=lambda x: x['hit_ratio'])
                print(f"Target {target_key}: 多个ref竞争，选择最佳 ref({best_match['ref_idx']},{best_match['ref_id']}) hit_ratio={best_match['hit_ratio']:.3f}")
                final_matches.append(best_match)
            else:
                final_matches.append(target_matches[0])
        
        print(f"过滤前匹配数: {len(matches)}")
        print(f"过滤后匹配数: {len(final_matches)}")
        
        return final_matches
    
    def analyze_with_filtering(self) -> Dict:
        """分析所有匹配结果并应用过滤"""
        print("\n=== 分析匹配结果（带过滤）===")
        
        all_matched_pairs = []
        
        # 收集所有匹配（动态发现参考索引目录）
        ref_indices = []
        if self.summary_dir.exists():
            for p in sorted(self.summary_dir.iterdir(), key=lambda x: x.name):
                if p.is_dir() and p.name.isdigit():
                    ref_indices.append(int(p.name))
        
        for ref_idx in ref_indices:
            summary_file = self.summary_dir / str(ref_idx) / "matching_summary.txt"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                match_pattern = r'Matched ref (\d+) → target (\d+) \(hit ratio: ([\d.]+) (\d+)/(\d+)\)'
                matches = re.findall(match_pattern, content)
                target_image_pattern = r'target image (\d+)'
                
                current_target_img = None
                match_idx = 0
                
                for line in content.split('\n'):
                    target_match = re.search(target_image_pattern, line)
                    if target_match:
                        current_target_img = int(target_match.group(1))
                    elif 'Matched ref' in line and match_idx < len(matches):
                        match = matches[match_idx]
                        match_info = {
                            'ref_idx': ref_idx,
                            'ref_id': int(match[0]),
                            'target_idx': current_target_img,
                            'target_id': int(match[1]),
                            'hit_ratio': float(match[2]),
                            'matched_points': int(match[3]),
                            'total_points': int(match[4])
                        }
                        all_matched_pairs.append(match_info)
                        match_idx += 1
        
        # 应用最佳匹配过滤
        filtered_matches = self.filter_best_matches(all_matched_pairs)
        
        return {
            'original_matches': len(all_matched_pairs),
            'filtered_matches': len(filtered_matches),
            'pairs': filtered_matches
        }

def main():
    """演示改进算法"""
    analyzer = ImprovedSKUCountAnalyzer("../imdata/detections_results", "output_pt")
    result = analyzer.analyze_with_filtering()
    
    print(f"\n原始匹配数: {result['original_matches']}")
    print(f"过滤后匹配数: {result['filtered_matches']}")
    print(f"减少了: {result['original_matches'] - result['filtered_matches']} 个冗余匹配")

if __name__ == "__main__":
    main()
