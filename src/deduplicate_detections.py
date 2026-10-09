"""
根据匹配关系对检出框去重的实用脚本
uv run python src/deduplicate_detections.py --dataset imdata/floor_display2

用途
- 读取数据集的检测结果（detections_results/ 数字命名的 JSON）
- 解析 DA3 产出的 matching_summary.txt（参考/目标均为源图片文件 ID）
- 根据匹配边构建带同帧冲突约束的全局映射
- 按全局映射中的 removed 标志保留逐帧 JSON 对象

去重策略
- 按 hit ratio 从高到低合并匹配边；会造成同一图片多个对象进入同一组件的边将被跳过
- 每个最终连通分量中最早出现的对象标记为 removed=false，其余对象标记为 removed=true
- 逐帧去重 JSON 只保留 removed=false 的对象

输出
- 将去重后的 JSON 写入 <output_root>/<dataset_name>/dedup_detections/。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 添加父目录到路径以便导入utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.detection_objects import flatten_detection_objects
from utils.classification_aggregation import validate_classification


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    detections_dir: Path


def resolve_dataset_paths(
    dataset_dir: Path, detections_dir: Path | None = None
) -> DatasetPaths:
    detections_dir = detections_dir or dataset_dir / 'detections_results'
    return DatasetPaths(dataset_dir=dataset_dir, detections_dir=detections_dir)


def load_detection_objects(json_path: Path) -> Tuple[Dict, List[Dict]]:
    """读取检测 JSON，返回原始字典与 objects 列表（不改变顺序）。
    支持两种结构：
    - {"skus": [{"classes": {...}, "objects": [...]}]}
    - {"classes": {...}, "objects": [...]} 或 [ {"classes":..., "objects":...} ]
    """
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    return data, flatten_detection_objects(data)


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
    - 不依赖组头行的先后顺序，规避"匹配行在前、组头在后"的歧义
    - 只保留 ref_idx < target_idx 的匹配，避免重复

    返回列表元素示例：
      { 'ref_idx': 0, 'ref_id': 12, 'target_idx': 2, 'target_id': 7,
        'hit_ratio': 0.85, 'matched_points': 34, 'total_points': 40 }
    注：ref_idx/target_idx 为源图片文件 ID，与 matching_summary 组头一致。
    """
    matches_all: List[Dict] = []
    if not summary_root.exists():
        return matches_all

    # 允许保留“最后一帧 -> 第一帧”的环形配对（仅一条反向边）
    allow_reverse_pairs: Set[Tuple[int, int]] = set()
    re_group = re.compile(r"^Matching objects between reference image (\d+) and target image (\d+)")
    summaries = {
        p: (p / "matching_summary.txt").read_text(encoding="utf-8").splitlines()
        for p in summary_root.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "matching_summary.txt").is_file()
    }
    ref_indices = sorted({
        int(value) for lines in summaries.values() for line in lines
        if (match := re_group.search(line)) is not None
        for value in match.groups()
    })
    if len(ref_indices) >= 2:
        allow_reverse_pairs.add((ref_indices[-1], ref_indices[0]))

    re_match = re.compile(r"Matched ref\s+(\d+)\s*.*?target\s+(\d+)\s+\(hit ratio:\s+([\d.]+)\s+(\d+)/(\d+)\)")
    re_found = re.compile(r"^Found\s+(\d+)\s+matches\s+in\s+image\s+(\d+)")

    for p, lines in sorted(summaries.items(), key=lambda item: item[0].name):
        ref_idx = None
        group_target = None
        sf = p / "matching_summary.txt"

        # 缓存本文件中尚未分配 target 的匹配行
        pending: List[Tuple[int, int, float, int, int]] = []  # (ref_id, target_id, hit_ratio, mp, tp)

        for line in lines:
            group = re_group.search(line)
            if group:
                ref_idx, group_target = map(int, group.groups())
                continue
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
                if ref_idx is None or group_target != tgt_idx:
                    raise ValueError(f"{sf}: match group must identify its source and target file IDs")
                if len(pending) != n:
                    raise ValueError(f"{sf}: expected {n} matches, received {len(pending)}")

                # 只保留 ref_idx < target_idx 的匹配，避免重复
                # 但允许保留“最后一帧 -> 第一帧”的环形配对
                if ref_idx >= tgt_idx and (ref_idx, tgt_idx) not in allow_reverse_pairs:
                    pending = []  # 清空pending，跳过这些匹配
                    continue

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

        if pending:
            raise ValueError(f"{sf}: unmatched summary rows lack a target group")

    return matches_all


def filter_best_matches(matches: List[Dict]) -> List[Dict]:
    """对每对图片之间的匹配进行严格一对一过滤。

    确保在每个图片对(ref_idx, target_idx)中：
    - 每个ref_id只出现一次
    - 每个target_id只出现一次

    使用贪心算法：按hit_ratio降序排列，依次选择不冲突的匹配。
    """
    # 按图片对分组：{(ref_idx, target_idx): [matches]}
    pair_groups: Dict[Tuple[int, int], List[Dict]] = {}
    for m in matches:
        key = (m['ref_idx'], m['target_idx'])
        pair_groups.setdefault(key, []).append(m)

    final_matches: List[Dict] = []

    # 对每一对图片单独进行严格一对一过滤
    for (ref_idx, target_idx), pair_matches in pair_groups.items():
        # 按hit_ratio降序排列
        sorted_matches = sorted(pair_matches, key=lambda x: x['hit_ratio'], reverse=True)

        # 贪心选择：已使用的ref和target
        used_refs: Set[int] = set()
        used_targets: Set[int] = set()

        for m in sorted_matches:
            ref_id = m['ref_id']
            target_id = m['target_id']
            # 只有当ref和target都未被使用时才选择
            if ref_id not in used_refs and target_id not in used_targets:
                final_matches.append(m)
                used_refs.add(ref_id)
                used_targets.add(target_id)

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
                         dedup_mode: str = 'any', min_hit_ratio: float = 0.4,
                         output_subdir: str = None, algorithm: str = 'point_tracking',
                         backend: str | None = None,
                         detections_dir: Path | None = None) -> Dict[int, Path]:
    """根据最终全局映射的 removed 标志生成逐帧去重 JSON。

    每个映射连通分量中最早出现的对象保留，其余观察标记为删除；
    同帧冲突导致被跳过的匹配边不会单独删除对象。
    返回 {image_idx: 输出路径}。

    Args:
        paths: 数据集路径配置
        output_root: 输出根目录
        max_image: 最大处理图片数
        same_names: 是否使用相同文件名（否则添加_dedup后缀）
        dedup_mode: 去重模式 'any'/'best'
        min_hit_ratio: 最小命中率阈值
        output_subdir: 输出子目录名（如'dedup_detections'），若为None则直接输出到dataset_name/
        algorithm: 算法类型 'point_tracking'/'3d_mapping'，决定匹配结果目录名
        backend: 3D算法后端 'vggt'/'pi3'/'da3'，仅在algorithm='3d_mapping'时生效
    """
    if output_root is None:
        raise ValueError("output_root 不能为空，请指定匹配结果所在的输出根目录")

    if detections_dir is not None:
        paths = resolve_dataset_paths(paths.dataset_dir, detections_dir)

    dataset_name = paths.dataset_dir.name
    match_base = output_root / dataset_name

    # 根据算法类型和后端动态计算匹配结果目录
    if algorithm == 'point_tracking':
        summary_root = match_base / 'output_pt'
    elif algorithm in ('3d_mapping', '3d'):
        if backend:
            summary_root = match_base / f'output_3dmapping_{backend}'
        else:
            summary_root = match_base / 'output_3dmapping'
    else:
        raise ValueError(f"未知算法类型 '{algorithm}'")

    all_matches = parse_all_matches(summary_root)

    # 先读取检测文件编号，用于自动判断索引是否需要偏移
    indices = _list_numeric_detection_indices(paths.detections_dir)
    if not indices:
        raise FileNotFoundError(f"No detection JSON found in {paths.detections_dir}")

    det_set = set(indices)
    for match in all_matches:
        if match["ref_idx"] not in det_set or match["target_idx"] not in det_set:
            raise ValueError("matching summary file IDs do not match detection files; rerun matching")

    if min_hit_ratio > 0 and all_matches:
        all_matches = [m for m in all_matches if float(m.get('hit_ratio', 0.0)) >= min_hit_ratio]

    # 'any' 使用所有通过阈值的匹配边；'best' 先按图片对做一对一过滤。
    # 两种模式均由最终全局映射的 removed 标志决定逐帧保留项，
    # 被并查集拒绝的冲突边不会导致其端点被删除。
    if dedup_mode == 'best':
        matches_for_dedup = filter_best_matches(all_matches) if all_matches else []
    else:
        matches_for_dedup = all_matches
    matches_for_gid: List[Dict] = matches_for_dedup

    # === 调试：检查去重和全局ID使用的匹配数量差异 ===
    logger.info(f"[DEBUG] 去重使用的匹配数(matches_for_dedup): {len(matches_for_dedup)}")
    logger.info(f"[DEBUG] 全局ID使用的匹配数(matches_for_gid): {len(matches_for_gid)}")

    if max_image is not None:
        indices = [i for i in indices if i <= max_image]
    if not indices or indices[0] != 1:
        logger.warning("Detections do not start at 1; proceeding with available indices")

    # 输出目录：output_root/dataset_name 或 output_root/dataset_name/output_subdir
    if output_subdir:
        out_dir = output_root / dataset_name / output_subdir
    else:
        out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[int, Path] = {}
    # 最终保留集合从全局映射派生，避免与冲突边处理使用不同的去重决策。
    survivors_by_image: Dict[int, Set[int]] = {}
    objects_by_image: Dict[int, List[Dict]] = {}
    originals_by_image: Dict[int, Dict] = {}

    for i in indices:
        src = paths.detections_dir / f"{i}.json"
        if not src.exists():
            logger.warning(f"Detection JSON missing for image {i}: {src}")
            continue
        try:
            original, objects = load_detection_objects(src)
        except (OSError, ValueError) as error:
            logger.warning(f"Image {i}: 读取检测JSON失败，跳过（{error}）")
            continue

        originals_by_image[i] = original
        objects_by_image[i] = objects

    mapping_path = out_dir / 'global_mapping.json'
    global_skus_path = out_dir / 'global_skus.json'
    publication_paths = (mapping_path, global_skus_path)
    _remove_global_publication_files(publication_paths)
    try:
        # 所有内容先在内存中构建、校验；不得在得到完整 pair 前发布 direct 文件。
        mapping = build_global_mapping(
            matches_for_gid, survivors_by_image, objects_by_image, indices
        )
        _validate_global_mapping_for_publication(mapping)
        for entries in mapping.values():
            for entry in entries:
                if not entry.get('removed', False):
                    survivors_by_image.setdefault(entry['image_id'], set()).add(
                        entry['object_id']
                    )
        json_strings = add_global_id_to_jsons(
            detections_dir=paths.detections_dir,
            global_mapping=mapping,
            indices=indices,
        )
        global_skus = _parse_global_skus_for_publication(json_strings)

        # Per-frame output and global artifacts now share the same retained set.
        for i, objects in objects_by_image.items():
            dst = out_dir / (f"{i}.json" if same_names else f"{i}_dedup.json")
            try:
                save_detection_objects(
                    dst, originals_by_image[i],
                    [
                        obj for index, obj in enumerate(objects)
                        if index in survivors_by_image.get(i, set())
                    ],
                )
            except Exception as error:  # noqa: BLE001 - 单图写入失败不影响整批去重
                logger.warning(f"Image {i}: 逐图去重JSON写入失败，跳过（{error}）")
                continue
            outputs[i] = dst

        _publish_global_pair(
            mapping_path=mapping_path,
            mapping=mapping,
            global_skus_path=global_skus_path,
            global_skus=global_skus,
        )
    except BaseException:
        _remove_global_publication_files(publication_paths)
        raise
    logger.info(f"Saved global mapping to: {mapping_path}")
    logger.info(f"Saved global SKUs with metadata to: {global_skus_path} ({len(global_skus)})")
    logger.info(f"Sequence dedup finished for images {indices[0]}..{indices[-1]} (total {len(indices)})")

    return outputs


def _validate_global_mapping_for_publication(mapping: Dict[str, List[Dict]]) -> None:
    if not isinstance(mapping, dict):
        raise ValueError("global mapping must be an object")
    for global_id, entries in mapping.items():
        if not isinstance(global_id, str) or not isinstance(entries, list):
            raise ValueError("global mapping entries are invalid")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("global mapping observation is invalid")
            validate_classification(entry.get("classification"))


def _parse_global_skus_for_publication(json_strings: List[str]) -> List[str]:
    if not isinstance(json_strings, list):
        raise ValueError("global SKUs must be a list")
    for item in json_strings:
        if not isinstance(item, str):
            raise ValueError("global SKU entry must be JSON text")
        value = json.loads(item, parse_constant=_reject_nonfinite_json_constant)
        if not isinstance(value, dict):
            raise ValueError("global SKU entry must decode to an object")
    return json_strings


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"global SKU JSON constant is invalid: {value}")


def _write_global_publication_temp(path: Path, payload: object) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_global_pair(
    *,
    mapping_path: Path,
    mapping: Dict[str, List[Dict]],
    global_skus_path: Path,
    global_skus: List[str],
) -> None:
    temporary_paths: List[Path] = []
    try:
        mapping_temporary = _write_global_publication_temp(mapping_path, mapping)
        temporary_paths.append(mapping_temporary)
        skus_temporary = _write_global_publication_temp(global_skus_path, global_skus)
        temporary_paths.append(skus_temporary)
        mapping_temporary.replace(mapping_path)
        temporary_paths.remove(mapping_temporary)
        skus_temporary.replace(global_skus_path)
        temporary_paths.remove(skus_temporary)
    except BaseException:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise


def _remove_global_publication_files(paths: Tuple[Path, Path]) -> None:
    errors: List[OSError] = []
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            errors.append(error)
    if errors:
        raise errors[0]


def build_global_mapping(
    matches: List[Dict],
    survivors_by_image: Dict[int, Set[int]],
    objects_by_image: Dict[int, List[Dict]],
    image_indices: List[int],
) -> Dict[str, List[Dict]]:
    """根据匹配关系构建全局映射，并标记每个组件中最早的保留观察。

    规则
    - 节点: (源图片文件 ID, 原始 object_idx)，包含全部原始检测
    - 边: 使用可信匹配连接两节点；拒绝同一图片不同对象被并入同一组件
    - 组件: 使用并查集聚类；按图像顺序和对象索引顺序为首次出现的组件分配自增ID(从1开始)
    - 保留: 每个组件中最早的观察标记为 removed=false，其余标记为 removed=true
    - 值: 每个全局ID下是多个子字典，包含 image_id、object_id、bbox 等原始信息
    """
    # 按置信度降序：高置信边先合并，使连通分量锚定在强匹配上，避免弱错误边污染聚类
    matches = sorted(matches, key=lambda m: float(m.get('hit_ratio', 0.0)), reverse=True)

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
        r_img = int(m.get('ref_idx', -1))  # 保持0-based，与文件编号一致
        t_img = int(m.get('target_idx', -1))
        r_id = int(m.get('ref_id', -1))
        t_id = int(m.get('target_id', -1))
        nodes_with_matches.add((r_img, r_id))
        nodes_with_matches.add((t_img, t_id))

    # === 调试信息 ===
    logger.info(f"[DEBUG] 匹配关系数量: {len(matches)}")
    logger.info(f"[DEBUG] 有匹配关系的节点数: {len(nodes_with_matches)}")

    # 初始化全部原始检测，未连接的检测保留独立组件。
    for img_id in image_indices:
        objs = objects_by_image.get(img_id, [])
        for obj_idx in range(len(objs)):
            # 每个原始检测都必须进入全局映射；没有可信边的对象成为独立组件。
            parent[(img_id, obj_idx)] = (img_id, obj_idx)
            rank[(img_id, obj_idx)] = 0

    logger.info(f"[DEBUG] 初始化节点数(parent): {len(parent)}")

    # 添加边（使用所有匹配连接对应节点）
    # 记录每个连通分量包含哪些图片的物体: root -> {img_id: set(obj_ids)}
    comp_imgs: Dict[Tuple[int, int], Dict[int, Set[int]]] = {
        node: {node[0]: {node[1]}} for node in parent.keys()
    }

    union_count = 0
    skipped_count = 0
    for m in matches:
        r_img = int(m.get('ref_idx', -1))
        t_img = int(m.get('target_idx', -1))
        r_id = int(m.get('ref_id', -1))
        t_id = int(m.get('target_id', -1))
        r_node, t_node = (r_img, r_id), (t_img, t_id)
        if r_node not in parent or t_node not in parent:
            raise ValueError(f"Matching edge references an unknown detection: {r_node} -> {t_node}")
        ra, rb = find(r_node), find(t_node)
        if ra == rb:
            union_count += 1
            continue
        # 检查合并是否会导致同一图片内多个物体被合并
        conflict = any(img in comp_imgs.get(rb, {}) for img in comp_imgs.get(ra, {}))
        if conflict:
            skipped_count += 1
            continue
        # 执行union并合并映射
        union(r_node, t_node)
        new_root = find(r_node)
        old_root = rb if new_root == ra else ra
        for img, objs in comp_imgs.pop(old_root, {}).items():
            comp_imgs.setdefault(new_root, {}).setdefault(img, set()).update(objs)
        union_count += 1

    logger.info(f"[DEBUG] 执行union操作次数: {union_count}, 跳过冲突: {skipped_count}")

    # 统计连通分量数量
    unique_roots = set(find(node) for node in parent.keys())
    logger.info(f"[DEBUG] 连通分量数量(即全局ID数): {len(unique_roots)}")

    # 为每个组件分配全局ID（按遇到顺序）
    comp_to_gid: Dict[Tuple[int, int], int] = {}
    gid_counter = 1
    mapping: Dict[str, List[Dict]] = {}
    # 记录每个全局ID中最早出现的物体（用于间接去重）
    gid_first_occurrence: Dict[int, Tuple[int, int]] = {}

    # 分配全局ID（只为parent中的节点）
    for img_id in sorted(image_indices):
        objects = objects_by_image.get(img_id, [])
        for obj_idx in range(len(objects)):
            # 跳过不在parent中的节点
            if (img_id, obj_idx) not in parent:
                continue
            root = find((img_id, obj_idx))
            if root not in comp_to_gid:
                comp_to_gid[root] = gid_counter
                gid_first_occurrence[gid_counter] = (img_id, obj_idx)
                gid_counter += 1
            gid = comp_to_gid[root]
            key = str(gid)
            obj = objects[obj_idx]
            # 只有最早出现的物体标记为removed=False（间接去重）
            first_img, first_obj = gid_first_occurrence[gid]
            is_removed = (img_id, obj_idx) != (first_img, first_obj)
            entry = {
                'image_id': img_id,
                'object_id': obj_idx,
                'bbox': obj.get('position'),
                'removed': is_removed,
                'classification': deepcopy(
                    validate_classification(obj.get('classification'))
                ),
            }
            mapping.setdefault(key, []).append(entry)

    retained_by_image = {img_id: 0 for img_id in image_indices}
    for entries in mapping.values():
        for entry in entries:
            if not entry['removed']:
                retained_by_image[entry['image_id']] += 1
    for img_id in image_indices:
        logger.info(
            f"[DEBUG] 图片{img_id}: 原始物体={len(objects_by_image.get(img_id, []))}, "
            f"保留物体={retained_by_image[img_id]}"
        )
    retained_count = sum(retained_by_image.values())
    logger.info(f"[DEBUG] 保留物体总数: {retained_count}")
    logger.info(
        f"[DEBUG] 其中: 保留物体节点={retained_count}, "
        f"仅匹配节点(被删除)={len(parent) - retained_count}"
    )

    return mapping


def add_global_id_to_jsons(
    *,
    detections_dir: Path | None = None,
    global_mapping: Dict[str, List[Dict]],
    indices: List[int],
) -> List[str]:
    """为检测结果JSON添加global_id和is_deduplicated字段（保留所有原始检出框）。

    Args:
        detections_dir: 原始检测结果目录
        global_mapping: 全局ID映射 {global_id: [{image_id, object_id, bbox, removed}, ...]}
        indices: 图片索引列表

    Returns:
        JSON字符串列表，每个元素对应一张图片的检测结果
    """
    import copy

    if detections_dir is None:
        raise ValueError("detections_dir is required for global ID publication")

    json_strings: List[str] = []

    # 构建反向映射：{(image_id, object_id): (global_id, removed)}
    reverse_mapping: Dict[Tuple[int, int], Tuple[int, bool]] = {}
    for gid_str, entries in global_mapping.items():
        gid = int(gid_str)
        for entry in entries:
            img_id = entry['image_id']
            obj_id = entry['object_id']
            removed = entry.get('removed', False)
            reverse_mapping[(img_id, obj_id)] = (gid, removed)

    for i in indices:
        src = detections_dir / f"{i}.json"
        if not src.exists():
            logger.warning(f"Detection JSON missing for image {i}: {src}")
            continue

        # 读取原始检测结果
        original, objects = load_detection_objects(src)

        # 为每个object添加global_id和is_deduplicated字段
        objects_with_metadata = []
        for idx, obj in enumerate(objects):
            obj_copy = copy.deepcopy(obj)
            # 从reverse_mapping获取global_id和removed状态
            mapping_info = reverse_mapping.get((i, idx))
            if mapping_info:
                gid, removed = mapping_info
                obj_copy['global_id'] = gid
                obj_copy['is_deduplicated'] = removed
            else:
                obj_copy['global_id'] = None
                obj_copy['is_deduplicated'] = False
            objects_with_metadata.append(obj_copy)

        # 构造输出数据（扁平格式）
        if isinstance(original, dict) and 'skus' in original and isinstance(original['skus'], list) and original['skus']:
            data = {
                'classes': copy.deepcopy(original['skus'][0].get('classes', {})),
                'objects': objects_with_metadata
            }
        elif isinstance(original, list) and original:
            data = {
                'classes': copy.deepcopy(original[0].get('classes', {})),
                'objects': objects_with_metadata
            }
        elif isinstance(original, dict) and 'objects' in original:
            data = {
                'classes': copy.deepcopy(original.get('classes', {})),
                'objects': objects_with_metadata
            }
        else:
            logger.warning(f"Unsupported JSON structure for image {i}, skipping")
            continue

        # 将数据转换为JSON字符串并添加到列表
        json_str = json.dumps(data, ensure_ascii=False)
        json_strings.append(json_str)

        # 统计被去重的对象数量
        dedup_count = sum(1 for obj in objects_with_metadata if obj.get('is_deduplicated', False))
        logger.info(f"Image {i}: processed {len(objects_with_metadata)} objects "
                   f"({dedup_count} marked as deduplicated)")

    return json_strings


def main():
    import argparse

    parser = argparse.ArgumentParser(description='根据匹配关系对检出框序列去重')
    parser.add_argument('--dataset', type=str, default='imdata/floor_display2', help='数据集根目录')
    parser.add_argument('--max_image', type=int, default=None, help='处理到最大图片编号(含)')
    parser.add_argument('--output_root', type=str, default='Output', help='匹配输入与去重结果的输出根目录')
    parser.add_argument('--same_names', action='store_true', help='输出文件名与原始一致（1.json, 2.json, ...），而不是 *_dedup.json')
    parser.add_argument('--dedup_mode', type=str, choices=['any', 'best'], default='any',
                        help="去重策略：any=使用所有匹配；best=一对一过滤后再去重（默认 any）")
    parser.add_argument('--min_hit_ratio', type=float, default=0.0,
                        help='最小命中率过滤阈值（默认0不过滤，例如 0.6）')
    parser.add_argument('--algorithm', type=str, choices=['point_tracking', '3d', '3d_mapping'], default='3d',
                        help='算法类型：point_tracking=点追踪；3d=DA3 3D映射（默认 3d）')
    parser.add_argument('--backend', type=str, choices=['vggt', 'pi3', 'da3'], default='da3',
                        help='3D算法后端（默认 da3）')

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        candidate = Path(__file__).parent.parent / args.dataset
        if candidate.exists():
            dataset_dir = candidate
    paths = resolve_dataset_paths(dataset_dir)

    if not paths.detections_dir.exists():
        raise FileNotFoundError(f"Detections dir not found: {paths.detections_dir}")
    output_root = Path(args.output_root)

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
