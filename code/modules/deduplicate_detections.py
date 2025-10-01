"""
根据匹配关系对检出框去重的实用脚本
uv run python modules/deduplicate_detections.py --dataset imdata0911/floor_display2

用途
- 读取数据集的检测结果（detections_results/ 数字命名的 JSON）
- 解析 VGGT 产出的 matching_summary.txt（参考 0-based → 目标 0-based 的对应关系）
- 依据同一参考物体(ref_id)在早期图片中已出现的情况，去除指定图片中的重复检出框

默认策略（序列去重）
- 处理图片 1..N，其中图片 1 保持不变
- 对每张图片 i(i>1)，删除它与之前任一图片(1..i-1)中已出现 ref_id 的对应 target_id 的检出框

输出
- 将去重后的 JSON 写入 output_dedup/<dataset_name>/<image_idx>_dedup.json（或 --output_dir 指定的路径）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    detections_dir: Path
    summary_file: Path


def resolve_dataset_paths(dataset_dir: Path) -> DatasetPaths:
    detections_dir = dataset_dir / 'detections_results'
    summary_file = dataset_dir / 'output_pt' / '0' / 'matching_summary.txt'
    return DatasetPaths(dataset_dir=dataset_dir, detections_dir=detections_dir, summary_file=summary_file)


def load_detection_objects(json_path: Path) -> Tuple[Dict, List[Dict]]:
    """读取检测 JSON，返回原始字典与 objects 列表（不改变顺序）。
    支持两种结构：
    - {"skus": [{"classes": {...}, "objects": [...]}]}
    - {"classes": {...}, "objects": [...]} 或 [ {"classes":..., "objects":...} ]
    """
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # 归一化结构
    if isinstance(data, dict) and 'skus' in data:
        if isinstance(data['skus'], list) and data['skus']:
            node = data['skus'][0]
            if 'objects' not in node:
                raise ValueError(f"No 'objects' in {json_path}")
            return data, node['objects']
        raise ValueError(f"Empty 'skus' array in {json_path}")
    elif isinstance(data, list) and data:
        node = data[0]
        if 'objects' not in node:
            raise ValueError(f"No 'objects' in {json_path}")
        return data, node['objects']
    elif isinstance(data, dict) and 'objects' in data:
        return data, data['objects']
    else:
        raise ValueError(f"Unsupported detection JSON structure: {json_path}")


def save_detection_objects(json_path: Path, original: Dict, new_objects: List[Dict]) -> None:
    """将 new_objects 写回到原始结构中，并保存为 json_path。"""
    data = original
    if isinstance(data, dict) and 'skus' in data and isinstance(data['skus'], list) and data['skus']:
        data['skus'][0]['objects'] = new_objects
    elif isinstance(data, list) and data:
        data[0]['objects'] = new_objects
    elif isinstance(data, dict) and 'objects' in data:
        data['objects'] = new_objects
    else:
        raise ValueError("Unsupported detection JSON structure when saving")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved deduplicated detections to: {json_path}")


def parse_matching_summary(summary_path: Path) -> Dict[int, List[Tuple[int, int]]]:
    """解析 matching_summary.txt，返回 {actual_target_idx(1-based): [(ref_id, target_id), ...]}。

    说明：
    - 文件内索引为0-based：reference image <r0>, target image <t0>
    - 为了与检测文件 1.json/2.json/3.json 对齐，这里输出为 1-based 下标 actual_target=t0+1
    - 匹配详情行在组头行之前，因此需缓存后在组头处落盘
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"matching_summary.txt not found: {summary_path}")

    # 行格式示例: "Matched ref 56 → target 79 (hit ratio: 0.70 32/46)"
    # 使用鲁棒正则，避免直接依赖特殊箭头字符
    match_line_re = re.compile(r"^Matched ref (\d+).*?target\s+(\d+)\s+\(")
    group_header_re = re.compile(r"^Matching objects between reference image (\d+) and target image (\d+)")

    pairs_by_target: Dict[int, List[Tuple[int, int]]] = {}
    buffer: List[Tuple[int, int]] = []

    total_pairs = 0
    with summary_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = match_line_re.match(line)
            if m:
                ref_id = int(m.group(1))
                target_id = int(m.group(2))
                buffer.append((ref_id, target_id))
                total_pairs += 1
                continue
            # 回退解析：更宽松的提取（防止正则模式因特殊字符编码差异导致失配）
            if line.startswith("Matched ref ") and "target" in line:
                try:
                    m1 = re.search(r"Matched ref\s+(\d+)", line)
                    m2 = re.search(r"target\s+(\d+)", line)
                    if m1 and m2:
                        ref_id = int(m1.group(1))
                        target_id = int(m2.group(1))
                        buffer.append((ref_id, target_id))
                        total_pairs += 1
                        continue
                except Exception:
                    pass
            g = group_header_re.match(line)
            if g:
                # 刷新到对应 target
                t0 = int(g.group(2))
                actual_target = t0 + 1
                if buffer:
                    pairs_by_target.setdefault(actual_target, []).extend(buffer)
                    buffer = []

    if total_pairs == 0:
        logger.warning(f"No 'Matched ref ... target ...' pairs parsed in summary: {summary_path}")
    return pairs_by_target


def compute_dedup_indices(pairs_by_target: Dict[int, List[Tuple[int, int]]],
                          keep_images: Set[int], dedup_image: int) -> Set[int]:
    """根据对应关系，计算在 dedup_image 中需要去除的 target_id 集合（object_id）。

    规则：
    - 先汇总 keep_images 中出现过的所有 ref_id 集合 R_keep
    - 在 dedup_image 中，凡是 (ref_id, target_id) 的 ref_id ∈ R_keep 的，都将其 target_id 标记为去除
    """
    ref_keep: Set[int] = set()
    for img in keep_images:
        for ref_id, _t in pairs_by_target.get(img, []):
            ref_keep.add(ref_id)

    drop_ids: Set[int] = set()
    for ref_id, target_id in pairs_by_target.get(dedup_image, []):
        if ref_id in ref_keep:
            drop_ids.add(target_id)
    return drop_ids


def parse_all_matches(summary_root: Path) -> List[Dict]:
    """扫描 summary_root 下的所有参考目录，解析并返回完整的匹配列表。

    解析策略（鲁棒版）
    - 先缓存所有 "Matched ref ... → target ... (hit ratio ...)" 行为未分配列表
    - 当遇到 "Found N matches in image T" 时，将最近的 N 条缓存匹配分配给 target_idx=T（0-based）
    - 不依赖组头行的先后顺序，规避“匹配行在前、组头在后”的歧义

    返回列表元素示例：
      { 'ref_idx': 0, 'ref_id': 12, 'target_idx': 2, 'target_id': 7,
        'hit_ratio': 0.85, 'matched_points': 34, 'total_points': 40 }
    注：ref_idx/target_idx 为 0-based，与 matching_summary 一致。
    """
    matches_all: List[Dict] = []
    if not summary_root.exists():
        return matches_all

    re_match = re.compile(r"Matched ref\s+(\d+)\s*.*?target\s+(\d+)\s+\(hit ratio:\s+([\d.]+)\s+(\d+)/(\d+)\)")
    re_found = re.compile(r"^Found\s+(\d+)\s+matches\s+in\s+image\s+(\d+)")

    for p in sorted(summary_root.iterdir(), key=lambda x: x.name):
        if not (p.is_dir() and p.name.isdigit()):
            continue
        ref_idx = int(p.name)
        sf = p / "matching_summary.txt"
        if not sf.exists():
            continue
        try:
            lines = sf.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            lines = sf.read_text(errors='ignore').splitlines()

        # 缓存本文件中尚未分配 target 的匹配行
        pending: List[Tuple[int, int, float, int, int]] = []  # (ref_id, target_id, hit_ratio, mp, tp)

        for line in lines:
            m = re_match.search(line)
            if m:
                try:
                    pending.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5))))
                except Exception:
                    continue
                continue

            f = re_found.search(line)
            if f:
                try:
                    n = int(f.group(1))
                    tgt_idx = int(f.group(2))  # 0-based
                except Exception:
                    continue
                if n <= 0:
                    continue
                if len(pending) < n:
                    logger.warning(f"[{sf}] Found {n} but only {len(pending)} pending matches; assigning all pending")
                    n = len(pending)
                # 取最近的 n 条匹配行归属给该 target
                block = pending[-n:]
                pending = pending[:-n]
                for ref_id, target_id, hit_ratio, mp, tp in block:
                    matches_all.append({
                        'ref_idx': ref_idx,
                        'ref_id': ref_id,
                        'target_idx': tgt_idx,
                        'target_id': target_id,
                        'hit_ratio': hit_ratio,
                        'matched_points': mp,
                        'total_points': tp,
                    })

        # 若文件末尾仍有 pending 未分配（缺少 Found 行），则无法确定 target_idx，跳过

    return matches_all


def filter_best_matches(matches: List[Dict]) -> List[Dict]:
    """每个ref保留最佳target；每个target保留最佳ref。"""
    # 按 (ref_idx, ref_id) 分组
    ref_groups: Dict[Tuple[int, int], List[Dict]] = {}
    for m in matches:
        key = (m['ref_idx'], m['ref_id'])
        ref_groups.setdefault(key, []).append(m)
    ref_best: List[Dict] = []
    for _k, lst in ref_groups.items():
        ref_best.append(max(lst, key=lambda x: x['hit_ratio']))

    # 按 (target_idx, target_id) 分组，解决多个ref争夺同一target
    tgt_groups: Dict[Tuple[int, int], List[Dict]] = {}
    for m in ref_best:
        key = (m['target_idx'], m['target_id'])
        tgt_groups.setdefault(key, []).append(m)

    final_matches: List[Dict] = []
    for _k, lst in tgt_groups.items():
        if len(lst) == 1:
            final_matches.append(lst[0])
        else:
            final_matches.append(max(lst, key=lambda x: x['hit_ratio']))
    return final_matches


def _list_numeric_detection_indices(detections_dir: Path) -> List[int]:
    nums: List[int] = []
    for p in detections_dir.glob('*.json'):
        try:
            nums.append(int(p.stem))
        except ValueError:
            continue
    nums.sort()
    return nums


def deduplicate_for_images(paths: DatasetPaths,
                           keep_images: Set[int], dedup_image: int,
                           output_root: Path | None = None,
                           same_names: bool = False) -> Dict[int, Path]:
    """执行一次性去重（兼容旧用法）。返回 {image_idx: 输出路径}。image_idx 为 1-based。"""
    pairs_by_target = parse_matching_summary(paths.summary_file)
    to_drop_in_dedup = compute_dedup_indices(pairs_by_target, keep_images, dedup_image)
    logger.info(f"Dedup plan: drop {len(to_drop_in_dedup)} boxes in image {dedup_image}")

    dataset_name = paths.dataset_dir.name
    # 默认输出到代码根目录下的 output_dedup/<dataset_name>
    out_dir = (output_root or (Path(__file__).parent.parent / 'output_dedup')) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[int, Path] = {}
    # 保留图
    for img_idx in sorted(keep_images):
        src = paths.detections_dir / f"{img_idx}.json"
        if not src.exists():
            logger.warning(f"Detection JSON missing for image {img_idx}: {src}")
            continue
        original, objects = load_detection_objects(src)
        dst = out_dir / (f"{img_idx}.json" if same_names else f"{img_idx}_dedup.json")
        save_detection_objects(dst, original, objects)
        outputs[img_idx] = dst

    # 去重图
    src = paths.detections_dir / f"{dedup_image}.json"
    if not src.exists():
        logger.warning(f"Detection JSON missing for dedup image {dedup_image}: {src}")
    else:
        original, objects = load_detection_objects(src)
        filtered = [obj for idx, obj in enumerate(objects) if idx not in to_drop_in_dedup]
        dst = out_dir / (f"{dedup_image}.json" if same_names else f"{dedup_image}_dedup.json")
        save_detection_objects(dst, original, filtered)
        outputs[dedup_image] = dst

    return outputs


def deduplicate_sequence(paths: DatasetPaths, output_root: Path | None = None,
                         max_image: int | None = None, same_names: bool = False,
                         dedup_mode: str = 'any', min_hit_ratio: float = 0.0) -> Dict[int, Path]:
    """对 1..N 序列依次去重：
    - 第1张保留原样
    - 第i张(i>1)去除在 1..i-1 中已出现过（有匹配）的 ref_id 在第i张对应的 target_id。
    返回 {image_idx: 输出路径}。
    """
    # 解析所有参考索引的匹配
    summary_root = paths.dataset_dir / 'output_pt'
    all_matches = parse_all_matches(summary_root)

    # 选择去重策略：'any' 使用所有匹配；'best' 使用一对一过滤
    matches_for_dedup: List[Dict]
    if dedup_mode == 'best':
        matches_for_dedup = filter_best_matches(all_matches) if all_matches else []
    else:
        matches_for_dedup = all_matches

    # 命中率阈值过滤（可选）
    if min_hit_ratio > 0 and matches_for_dedup:
        matches_for_dedup = [m for m in matches_for_dedup if float(m.get('hit_ratio', 0.0)) >= min_hit_ratio]

    # 全局ID构建使用更保守的“一对一”匹配，避免将同一张图片的多个框误并为一个全局ID
    matches_for_gid: List[Dict] = filter_best_matches(all_matches) if all_matches else []

    # 建立两个删除集合映射（均按 0-based 图像索引记录待删对象ID）
    # 1) target 视角：删除 image t 中被任何前序参考图命中的 target_id（r_idx < t_idx）
    drop_target_map: Dict[int, Set[int]] = {}
    # 2) reference 视角：删除 image r 中那些与任何更早图像存在对应的 ref_id（t_idx < r_idx）
    drop_ref_map: Dict[int, Set[int]] = {}
    for m in matches_for_dedup:
        t_idx = m.get('target_idx')
        r_idx = m.get('ref_idx')
        if t_idx is None or r_idx is None:
            continue
        # 前序参考图 → 当前 target：删除当前图的 target_id
        if r_idx < t_idx:
            drop_target_map.setdefault(t_idx, set()).add(int(m['target_id']))
        # 当前参考图 → 更早 target：删除当前图的 ref_id
        if t_idx < r_idx:
            drop_ref_map.setdefault(r_idx, set()).add(int(m['ref_id']))
    indices = _list_numeric_detection_indices(paths.detections_dir)
    if not indices:
        raise FileNotFoundError(f"No detection JSON found in {paths.detections_dir}")
    if max_image is not None:
        indices = [i for i in indices if i <= max_image]
    if not indices or indices[0] != 1:
        logger.warning("Detections do not start at 1; proceeding with available indices")

    dataset_name = paths.dataset_dir.name
    # 默认输出到代码根目录下的 output_dedup/<dataset_name>
    out_dir = (output_root or (Path(__file__).parent.parent / 'output_dedup')) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[int, Path] = {}
    # 记录每张图保留下来的对象索引（基于原始 objects 下标）与其对象信息
    survivors_by_image: Dict[int, Set[int]] = {}
    objects_by_image: Dict[int, List[Dict]] = {}

    for i in indices:
        src = paths.detections_dir / f"{i}.json"
        if not src.exists():
            logger.warning(f"Detection JSON missing for image {i}: {src}")
            continue

        original, objects = load_detection_objects(src)

        if i == 1:
            # 保留原样
            dst = out_dir / (f"{i}.json" if same_names else f"{i}_dedup.json")
            save_detection_objects(dst, original, objects)
            outputs[i] = dst
            survivors_by_image[i] = set(range(len(objects)))
            objects_by_image[i] = objects
        else:
            t0 = i - 1  # 0-based index of this image
            drop_from_targets = drop_target_map.get(t0, set())
            drop_from_refs = drop_ref_map.get(t0, set())
            drop_ids = set(drop_from_targets) | set(drop_from_refs)
            new_objects = [obj for idx, obj in enumerate(objects) if idx not in drop_ids]
            survivors_by_image[i] = set(idx for idx in range(len(objects)) if idx not in drop_ids)
            objects_by_image[i] = objects
            logger.debug(
                f"Image {i}: drop {len(drop_ids)} boxes from {len(objects)} "
                f"(targets:{len(drop_from_targets)}, refs:{len(drop_from_refs)})"
            )
            dst = out_dir / (f"{i}.json" if same_names else f"{i}_dedup.json")
            save_detection_objects(dst, original, new_objects)
            outputs[i] = dst

    logger.info(f"Sequence dedup finished for images {indices[0]}..{indices[-1]} (total {len(indices)})")
    # 构建全局唯一ID映射（只针对去重后保留的对象）
    try:
        mapping = build_global_mapping(matches_for_gid, survivors_by_image, objects_by_image, indices)
        mapping_path = out_dir / 'global_mapping.json'
        with mapping_path.open('w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved global mapping to: {mapping_path}")
    except Exception as e:
        logger.warning(f"Failed to build/save global mapping: {e}")

    return outputs


def build_global_mapping(
    matches: List[Dict],
    survivors_by_image: Dict[int, Set[int]],
    objects_by_image: Dict[int, List[Dict]],
    image_indices: List[int],
) -> Dict[str, List[Dict]]:
    """根据匹配关系与去重结果，为保留的检出框分配全局唯一ID。

    规则
    - 节点: (image_id_1based, object_idx) 仅包含去重后保留的对象
    - 边: 使用匹配关系中连接的两节点（仅当两端都在保留集合中）
    - 组件: 使用并查集聚类；按图像顺序(1..N)和对象索引顺序为首次出现的组件分配自增ID(从1开始)
    - 值: 每个全局ID下是多个子字典，包含 image_id、object_id、bbox 等原始信息
    """
    # 并查集
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    rank: Dict[Tuple[int, int], int] = {}

    def find(x: Tuple[int, int]) -> Tuple[int, int]:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        rank.setdefault(ra, 0)
        rank.setdefault(rb, 0)
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # 初始化节点（所有对象：保留与被去重的都纳入全局图）
    for img_id in image_indices:
        objs = objects_by_image.get(img_id, [])
        for obj_idx in range(len(objs)):
            parent[(img_id, obj_idx)] = (img_id, obj_idx)
            rank[(img_id, obj_idx)] = 0

    # 添加边（使用所有匹配连接对应节点），以便被去重的节点仍隶属于其物体的全局组件
    for m in matches:
        r_img = int(m.get('ref_idx', -1)) + 1
        t_img = int(m.get('target_idx', -1)) + 1
        r_id = int(m.get('ref_id', -1))
        t_id = int(m.get('target_id', -1))
        if (r_img, r_id) in parent and (t_img, t_id) in parent:
            union((r_img, r_id), (t_img, t_id))

    # 为每个组件分配全局ID（按遇到顺序）
    comp_to_gid: Dict[Tuple[int, int], int] = {}
    gid_counter = 1
    mapping: Dict[str, List[Dict]] = {}

    # 分配全局ID并输出所有对象（包含被去重的，标注 removed=True/False）
    for img_id in sorted(image_indices):
        objects = objects_by_image.get(img_id, [])
        survivors = survivors_by_image.get(img_id, set())
        for obj_idx in range(len(objects)):
            root = find((img_id, obj_idx))
            if root not in comp_to_gid:
                comp_to_gid[root] = gid_counter
                gid_counter += 1
            gid = comp_to_gid[root]
            key = str(gid)
            obj = objects[obj_idx] if 0 <= obj_idx < len(objects) else {}
            entry = {
                'image_id': img_id,
                'object_id': obj_idx,
                'bbox': obj.get('position'),
                'removed': obj_idx not in survivors,
            }
            mapping.setdefault(key, []).append(entry)

    return mapping


def main():
    import argparse

    parser = argparse.ArgumentParser(description='根据匹配关系对检出框去重')
    parser.add_argument('--dataset', type=str, default=str(Path('imdata0911') / 'floor_display2'), help='数据集根目录')
    parser.add_argument('--summary', type=str, default=None, help='matching_summary.txt 路径（默认: <dataset>/output_pt/0/matching_summary.txt）')
    parser.add_argument('--keep', type=str, default=None, help='单次去重模式：保留原样的图片编号(1-based)，逗号分隔')
    parser.add_argument('--dedup', type=int, default=None, help='单次去重模式：需要去重的图片编号(1-based)')
    parser.add_argument('--max_image', type=int, default=None, help='序列去重模式：处理到最大图片编号(含)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出根目录（默认: output_dedup/<dataset_name>）')
    parser.add_argument('--same_names', action='store_true', help='输出文件名与原始一致（1.json, 2.json, ...），而不是 *_dedup.json')
    parser.add_argument('--dedup_mode', type=str, choices=['any', 'best'], default='any',
                        help="去重策略：any=使用所有匹配；best=一对一过滤后再去重（默认 any）")
    parser.add_argument('--min_hit_ratio', type=float, default=0.0,
                        help='最小命中率过滤阈值（默认0不过滤，例如 0.6）')

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        candidate = Path(__file__).parent.parent / args.dataset
        if candidate.exists():
            dataset_dir = candidate
    paths = resolve_dataset_paths(dataset_dir)
    if args.summary:
        paths = DatasetPaths(dataset_dir=paths.dataset_dir, detections_dir=paths.detections_dir, summary_file=Path(args.summary))

    if not paths.detections_dir.exists():
        raise FileNotFoundError(f"Detections dir not found: {paths.detections_dir}")
    if not paths.summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {paths.summary_file}")

    output_root = Path(args.output_dir) if args.output_dir else None

    # 模式选择：若提供了 --dedup（可选配 --keep），执行单次去重；否则执行序列去重 1..N
    if args.dedup is not None:
        keep_images: Set[int] = set()
        if args.keep:
            for tok in args.keep.split(','):
                tok = tok.strip()
                if tok:
                    keep_images.add(int(tok))
        else:
            logger.warning("--dedup 指定但未提供 --keep，将仅输出去重图并不复制保留图")

        outputs = deduplicate_for_images(paths, keep_images, args.dedup, output_root, same_names=args.same_names)
        logger.info("Dedup (single) completed. Outputs:")
        for k, v in outputs.items():
            logger.info(f"  image {k}: {v}")
    else:
        outputs = deduplicate_sequence(
            paths, output_root, args.max_image, same_names=args.same_names,
            dedup_mode=args.dedup_mode, min_hit_ratio=args.min_hit_ratio
        )
        logger.info("Dedup (sequence) completed. Outputs:")
        for k in sorted(outputs.keys()):
            logger.info(f"  image {k}: {outputs[k]}")


if __name__ == '__main__':
    main()
