#!/usr/bin/env python3
"""
改进的SKU计数分析器 - 处理一对多匹配问题
（位于根 `src/` 的 SKU 计数分析阶段）
"""

import re
from pathlib import Path
from typing import Dict, List


class ImprovedSKUCountAnalyzer:
    """改进的SKU计数分析器 - 解决过度聚类问题"""

    def __init__(self, detection_dir: str, summary_dir: str = "output_pt"):
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
            content = summary_file.read_text(encoding="utf-8")
            pending = []
            group_ref = None
            # visualization writes each target ID after that target's matching lines.
            for line in content.splitlines():
                group = re.fullmatch(
                    r"Matching objects between reference image (\d+) and target image (\d+)", line
                )
                if group:
                    group_ref = int(group[1])
                    continue
                match = re.fullmatch(
                    r"Matched ref (\d+) → target (\d+) \(hit ratio: ([\d.]+) (\d+)/(\d+)\)",
                    line,
                )
                if match:
                    pending.append(
                        {
                            "ref_idx": ref_idx,
                            "ref_id": int(match[1]),
                            "target_id": int(match[2]),
                            "hit_ratio": float(match[3]),
                            "matched_points": int(match[4]),
                            "total_points": int(match[5]),
                        }
                    )
                    continue
                block = re.fullmatch(r"Found (\d+) matches in image (\d+)", line)
                if block is None:
                    continue
                if len(pending) != int(block[1]):
                    raise ValueError(f"Matching block count mismatch: {summary_file}")
                if pending and group_ref is None:
                    raise ValueError(f"Matching block is missing reference file ID: {summary_file}")
                for pair in pending:
                    pair["ref_idx"] = group_ref
                    pair["target_idx"] = int(block[2])
                all_matched_pairs.extend(pending)
                pending = []
            if pending:
                raise ValueError(f"Matching block is missing target image: {summary_file}")

        from .deduplicate_detections import filter_best_matches as _dedup_filter

        filtered = _dedup_filter(all_matched_pairs)
        return {
            "original_matches": len(all_matched_pairs),
            "filtered_matches": len(filtered),
            "pairs": filtered,
        }


def main():
    analyzer = ImprovedSKUCountAnalyzer("../imdata/detections_results", "output_pt")
    res = analyzer.analyze_with_filtering()
    print(res)


if __name__ == "__main__":
    main()
