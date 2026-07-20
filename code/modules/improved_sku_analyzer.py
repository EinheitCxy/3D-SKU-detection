#!/usr/bin/env python3
"""
改进的SKU计数分析器 - 处理一对多匹配问题
（从 code/improved_sku_analyzer.py 移动至 modules）
"""

import re
from pathlib import Path
from typing import Dict, List


class ImprovedSKUCountAnalyzer:
    """改进的SKU计数分析器 - 解决过度聚类问题"""

    def __init__(self, detection_dir: str, summary_dir: str = "output_3dmapping_da3"):
        self.detection_dir = Path(detection_dir)
        self.summary_dir = Path(summary_dir)

    def analyze_with_filtering(self) -> Dict:
        """分析所有匹配结果并应用过滤"""
        all_matched_pairs: List[Dict] = []

        # 动态发现参考索引目录
        ref_indices: List[int] = []
        if self.summary_dir.exists():
            for p in sorted(self.summary_dir.iterdir(), key=lambda x: x.name):
                if p.is_dir() and p.name.isdigit():
                    ref_indices.append(int(p.name))

        for ref_idx in ref_indices:
            summary_file = self.summary_dir / str(ref_idx) / "matching_summary.txt"
            if not summary_file.exists():
                continue
            content = summary_file.read_text(encoding="utf-8", errors="ignore")

            match_pattern = (
                r"Matched ref (\d+) → target (\d+) \(hit ratio: ([\d.]+) (\d+)/(\d+)\)"
            )
            matches = re.findall(match_pattern, content)

            # 方案一：两遍解析法
            # 第一遍：收集所有目标图像映射信息
            target_mapping = {}
            for line in content.split("\n"):
                # 查找 "Matching objects between reference image X and target image Y"
                section_match = re.search(
                    r"reference image (\d+) and target image (\d+)", line
                )
                if section_match:
                    ref_img, target_img = int(section_match.group(1)), int(
                        section_match.group(2)
                    )
                    target_mapping[ref_img] = target_img

            # 第二遍：解析匹配行，使用映射关系确定target_idx
            for m in matches:
                ref_id = int(m[0])
                # 根据ref_id推断属于哪个参考图像（这里的逻辑需要根据实际情况调整）
                # 由于是从ref_idx目录下读取的，所以目标图像就是映射中对应的值
                target_idx = target_mapping.get(ref_idx, None)

                all_matched_pairs.append(
                    {
                        "ref_idx": ref_idx,
                        "ref_id": ref_id,
                        "target_idx": target_idx,
                        "target_id": int(m[1]),
                        "hit_ratio": float(m[2]),
                        "matched_points": int(m[3]),
                        "total_points": int(m[4]),
                    }
                )

        from .deduplicate_detections import filter_best_matches as _dedup_filter

        filtered = _dedup_filter(all_matched_pairs)
        return {
            "original_matches": len(all_matched_pairs),
            "filtered_matches": len(filtered),
            "pairs": filtered,
        }


def main():
    analyzer = ImprovedSKUCountAnalyzer(
        "../imdata/detections_results", "output_3dmapping_da3"
    )
    res = analyzer.analyze_with_filtering()
    print(res)


if __name__ == "__main__":
    main()
