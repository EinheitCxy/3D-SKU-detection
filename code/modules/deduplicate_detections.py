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
- 将去重后的 JSON 写入 output_dedup/<dataset_name>/<image_idx>.json（或 --output_dir 指定的路径）
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加父目录到路径以便导入utils
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    """将 new_objects 写回到扁平格式 {classes, objects} 中，并保存为 json_path。"""
    import copy

    # 提取classes信息，构造扁平格式
    if isinstance(original, dict) and 'skus' in original and isinstance(original['skus'], list) and original['skus']:
        # 从skus[0]中提取classes
        data = {
            'classes': copy.deepcopy(original['skus'][0].get('classes', {})),
            'objects': new_objects
        }
    elif isinstance(original, list) and original:
        # 已经是[{classes, objects}]格式
        data = {
            'classes': copy.deepcopy(original[0].get('classes', {})),
            'objects': new_objects
        }
    elif isinstance(original, dict) and 'objects' in original:
        # 已经是{classes, objects}格式
        data = {
            'classes': copy.deepcopy(original.get('classes', {})),
            'objects': new_objects
        }
    else:
        raise ValueError("Unsupported detection JSON structure when saving")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved deduplicated detections to: {json_path}")


def save_merged_detections_with_gid(json_path: Path,
                                    images_data: Dict[int, Tuple[List[Dict], List[int]]],
                                    originals_by_image: Dict[int, Dict]) -> None:
    """保存多张图片合并的带global_id的检出框JSON（每个图片数据先转为JSON字符串，再作为列表元素）。

    Args:
        json_path: 输出文件路径
        images_data: {image_id: (objects, global_ids)} 字典
        originals_by_image: {image_id: original_json_structure} 字典
    """
    import copy

    merged_output = []

    for image_id in sorted(images_data.keys()):
        objects, global_ids = images_data[image_id]
        if len(global_ids) != len(objects):
            logger.warning(f"Image {image_id}: global_ids length mismatch, skipping")
            continue

        # 为每个object添加global_id
        objects_with_gid = []
        for obj, gid in zip(objects, global_ids):
            obj_copy = copy.deepcopy(obj)
            obj_copy['global_id'] = gid
            objects_with_gid.append(obj_copy)

        # 获取原始JSON结构
        original = originals_by_image.get(image_id)
        if original is None:
            logger.warning(f"Image {image_id}: original structure not found, skipping")
            continue

        # 提取classes和objects，去掉skus层
        image_data = None
        if isinstance(original, dict) and 'skus' in original:
            if isinstance(original['skus'], list) and original['skus']:
                # 从skus[0]中提取classes，直接构造{classes, objects}
                image_data = {
                    'classes': copy.deepcopy(original['skus'][0].get('classes', {})),
                    'objects': objects_with_gid
                }
        elif isinstance(original, list) and original:
            # 已经是[{classes, objects}]格式
            image_data = {
                'classes': copy.deepcopy(original[0].get('classes', {})),
                'objects': objects_with_gid
            }
        elif isinstance(original, dict) and 'objects' in original:
            # 已经是{classes, objects}格式
            image_data = {
                'classes': copy.deepcopy(original.get('classes', {})),
                'objects': objects_with_gid
            }
        else:
            logger.warning(f"Image {image_id}: unsupported structure, skipping")
            continue

        # 将每个图片数据转换为JSON字符串后添加到列表
        image_data_str = json.dumps(image_data, ensure_ascii=False)
        merged_output.append(image_data_str)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(merged_output, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved merged detections with global_id to: {json_path} ({len(merged_output)} images)")


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
        except (FileNotFoundError, UnicodeDecodeError):
            lines = sf.read_text(errors='ignore').splitlines()

        # 缓存本文件中尚未分配 target 的匹配行
        pending: List[Tuple[int, int, float, int, int]] = []  # (ref_id, target_id, hit_ratio, mp, tp)

        for line in lines:
            m = re_match.search(line)
            if m:
                try:
                    pending.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5))))
                except (ValueError, IndexError):
                    continue
                continue

            f = re_found.search(line)
            if f:
                try:
                    n = int(f.group(1))
                    tgt_idx = int(f.group(2))  # 0-based
                except (ValueError, IndexError):
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
    """获取检测目录中所有有效的数字索引。

    Returns:
        排序后的文件索引列表
    """
    from utils.data_utils import load_detections

    try:
        detections_with_index = load_detections(str(detections_dir), return_index_map=True)
        return sorted([idx for idx, _ in detections_with_index])
    except (FileNotFoundError, ValueError) as e:
        logger.debug(f"Failed to load detections from {detections_dir}: {e}")
        return []


def deduplicate_sequence(paths: DatasetPaths, output_root: Path | None = None,
                         max_image: int | None = None, same_names: bool = False,
                         dedup_mode: str = 'any', min_hit_ratio: float = 0.0,
                         output_subdir: str = None, algorithm: str = 'point_tracking',
                         backend: str | None = None) -> Dict[int, Path]:
    """对 1..N 序列依次去重：
    - 第1张保留原样
    - 第i张(i>1)去除在 1..i-1 中已出现过（有匹配）的 ref_id 在第i张对应的 target_id。
    返回 {image_idx: 输出路径}。

    Args:
        paths: 数据集路径配置
        output_root: 输出根目录
        max_image: 最大处理图片数
        same_names: 是否使用相同文件名（否则添加_dedup后缀）
        dedup_mode: 去重模式 'any'/'best'
        min_hit_ratio: 最小命中率阈值
        output_subdir: 输出子目录名（如'dedup_detections'），若为None则直接输出到dataset_name/
        algorithm: 算法类型 'point_tracking'/'3d_projection'，决定匹配结果目录名
        backend: 3D算法后端 'vggt'/'pi3'，仅在algorithm='3d_projection'时生效
    """
    # 根据算法类型和后端动态计算匹配结果目录
    if algorithm == 'point_tracking':
        summary_root = paths.dataset_dir / 'output_pt'
    elif algorithm == '3d_projection':
        if backend:
            summary_root = paths.dataset_dir / f'output_3dmapping_{backend}'
        else:
            summary_root = paths.dataset_dir / 'output_3dmapping'
    else:
        # 默认回退到 output_pt（向后兼容）
        logger.warning(f"未知算法类型 '{algorithm}'，回退到 'output_pt'")
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
    # 如果指定了 output_subdir，则追加子目录
    if output_subdir:
        out_dir = (output_root or (Path(__file__).parent.parent / 'output_dedup')) / dataset_name / output_subdir
    else:
        out_dir = (output_root or (Path(__file__).parent.parent / 'output_dedup')) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[int, Path] = {}
    # 记录每张图保留下来的对象索引（基于原始 objects 下标）与其对象信息
    survivors_by_image: Dict[int, Set[int]] = {}
    objects_by_image: Dict[int, List[Dict]] = {}
    originals_by_image: Dict[int, Dict] = {}

    for i in indices:
        src = paths.detections_dir / f"{i}.json"
        if not src.exists():
            logger.warning(f"Detection JSON missing for image {i}: {src}")
            continue

        original, objects = load_detection_objects(src)
        originals_by_image[i] = original

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

        # 生成带global_id的合并JSON
        images_data_for_merge: Dict[int, Tuple[List[Dict], List[int]]] = {}
        for img_id in indices:
            survivors = survivors_by_image.get(img_id, set())
            objects = objects_by_image.get(img_id, [])
            kept_objects = [objects[idx] for idx in sorted(survivors) if idx < len(objects)]
            kept_gids = []
            for idx in sorted(survivors):
                if idx < len(objects):
                    # 从mapping中查找global_id
                    gid = None
                    for gid_str, entries in mapping.items():
                        for entry in entries:
                            if entry['image_id'] == img_id and entry['object_id'] == idx and not entry['removed']:
                                gid = int(gid_str)
                                break
                        if gid:
                            break
                    if gid:
                        kept_gids.append(gid)
            images_data_for_merge[img_id] = (kept_objects, kept_gids)

        merged_gid_path = out_dir / 'all_images_with_global_id.json'
        save_merged_detections_with_gid(merged_gid_path, images_data_for_merge, originals_by_image)
    except (ValueError, json.JSONEncodeError, FileNotFoundError, PermissionError) as e:
        logger.warning(f"Failed to build/save global mapping or global_id JSONs: {e}")

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

    # 先收集所有有匹配关系的节点
    nodes_with_matches = set()
    for m in matches:
        r_img = int(m.get('ref_idx', -1)) + 1
        t_img = int(m.get('target_idx', -1)) + 1
        r_id = int(m.get('ref_id', -1))
        t_id = int(m.get('target_id', -1))
        nodes_with_matches.add((r_img, r_id))
        nodes_with_matches.add((t_img, t_id))

    # 初始化节点（保留的对象 + 有匹配关系的对象）
    for img_id in image_indices:
        objs = objects_by_image.get(img_id, [])
        survivors = survivors_by_image.get(img_id, set())
        for obj_idx in range(len(objs)):
            # 保留的对象或有匹配关系的对象
            if obj_idx in survivors or (img_id, obj_idx) in nodes_with_matches:
                parent[(img_id, obj_idx)] = (img_id, obj_idx)
                rank[(img_id, obj_idx)] = 0

    # 添加边（使用所有匹配连接对应节点）
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

    # 分配全局ID（只为parent中的节点）
    for img_id in sorted(image_indices):
        objects = objects_by_image.get(img_id, [])
        survivors = survivors_by_image.get(img_id, set())
        for obj_idx in range(len(objects)):
            # 跳过不在parent中的节点
            if (img_id, obj_idx) not in parent:
                continue
            root = find((img_id, obj_idx))
            if root not in comp_to_gid:
                comp_to_gid[root] = gid_counter
                gid_counter += 1
            gid = comp_to_gid[root]
            key = str(gid)
            obj = objects[obj_idx]
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

    parser = argparse.ArgumentParser(description='根据匹配关系对检出框序列去重')
    parser.add_argument('--dataset', type=str, default=str(Path('imdata0911') / 'floor_display2'), help='数据集根目录')
    parser.add_argument('--summary', type=str, default=None, help='matching_summary.txt 路径（默认: <dataset>/output_pt/0/matching_summary.txt）')
    parser.add_argument('--max_image', type=int, default=None, help='处理到最大图片编号(含)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出根目录（默认: output_dedup/<dataset_name>）')
    parser.add_argument('--same_names', action='store_true', help='输出文件名与原始一致（1.json, 2.json, ...），而不是 *_dedup.json')
    parser.add_argument('--dedup_mode', type=str, choices=['any', 'best'], default='any',
                        help="去重策略：any=使用所有匹配；best=一对一过滤后再去重（默认 any）")
    parser.add_argument('--min_hit_ratio', type=float, default=0.0,
                        help='最小命中率过滤阈值（默认0不过滤，例如 0.6）')
    parser.add_argument('--algorithm', type=str, choices=['point_tracking', '3d_projection'], default='point_tracking',
                        help='算法类型：point_tracking=点追踪；3d_projection=3D投影（默认 point_tracking）')
    parser.add_argument('--backend', type=str, choices=['vggt', 'pi3'], default=None,
                        help='3D算法后端：vggt或pi3（仅在algorithm=3d_projection时生效）')

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

    # 执行序列去重 1..N
    outputs = deduplicate_sequence(
        paths, output_root, args.max_image, same_names=args.same_names,
        dedup_mode=args.dedup_mode, min_hit_ratio=args.min_hit_ratio,
        algorithm=args.algorithm, backend=args.backend
    )
    logger.info("Dedup (sequence) completed. Outputs:")
    for k in sorted(outputs.keys()):
        logger.info(f"  image {k}: {outputs[k]}")


if __name__ == '__main__':
    main()
