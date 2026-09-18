"""牌面面积计算流水线阶段。

从 da3_cache 加载 world_points（metric 世界坐标系），结合 global_mapping.json
的 global_id 标注，为每个去重后的物理商品计算 front-facing 投影面积（m²）
与牌面占比（Share of Shelf）。

前置依赖
--------
1. DA3 重建缓存：``<save_root>/<dataset>/da3_cache/predictions.npz``
2. 顺序去重输出：``<save_root>/<dataset>/dedup_detections/global_mapping.json``

可独立运行：``uv run python main.py --mode area --dataset ../imdata/floor_display2``

对齐说明
--------
- DA3 ``world_points`` 的第 0 维按 ``image_ids``（文件编号）排序；
  detection JSON 与 global_mapping.json 的 ``image_id`` 均为文件编号（0-based），
  三者天然对齐。
- bbox（原图坐标）-> world_points 网格坐标：用 ``Pi3ImageTransform``
  以"原图 PIL 尺寸 -> cache 的 (W,H)"构造纯 resize 映射，与 matching 阶段
  （sms.py: im.resize((TW,TH))）一致，无需依赖 process_res 推断。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from utils.bbox_3d_extractor import _flatten_objects, extract_3d_from_bboxes
from utils.facing_area import camera_view_normals, compute_facing_area_report
from utils.transforms import Pi3ImageTransform

logger = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _resolve_image_path(images_dir: Path, image_id: int) -> Path:
    """根据文件编号在 images 目录找到对应图像（扩展名自动匹配）。"""
    for ext in _IMG_EXTS:
        for cand in (images_dir / f"{image_id}{ext}", images_dir / f"{image_id}{ext.upper()}"):
            if cand.exists():
                return cand
    for p in images_dir.iterdir():
        if p.is_file() and p.stem == str(image_id):
            return p
    raise FileNotFoundError(f"未找到 image_id={image_id} 的图像，目录: {images_dir}")


def _extract_sku_class(det: Dict) -> str:
    """提取 detection 的 sku_class（skus[0].classes.det[0]，与 dedup load_detection_objects 一致）。"""
    skus = det.get("skus", []) if isinstance(det, dict) else []
    if skus and isinstance(skus[0], dict):
        cls = skus[0].get("classes", {}).get("det")
        if isinstance(cls, list) and cls:
            return str(cls[0])
        if cls is not None:
            return str(cls)
    return "unknown"


def run_facing_area(
    dataset_path: str,
    save_root: Path,
    backend: str = "da3",
    conf_threshold: float = 0.1,
    min_points: int = 3,
    bbox_margin: int = 0,
) -> Dict:
    """计算牌面面积并写报告 + 回写 global_mapping.json。

    Args:
        dataset_path: 数据集路径（含 images/ + detections_results/）
        save_root: 输出根目录（通常为 Output/）
        backend: 3D 重建后端（仅 da3）
        conf_threshold: world_points_conf 置信度阈值
        min_points: 单商品最少点数（少于此数 area=0）
        bbox_margin: bbox 在 world_points 网格坐标内缩像素

    Returns:
        {"success": bool, "report_path": str, "global_mapping_path": str, "summary": dict}
    """
    dataset_dir = Path(dataset_path)
    dataset_name = dataset_dir.name
    images_dir = dataset_dir / "images"
    detections_dir = dataset_dir / "detections_results"

    # 1. 定位 da3 cache
    cache_path = save_root / dataset_name / f"{backend}_cache" / "predictions.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"DA3 缓存不存在: {cache_path}，请先运行 --mode pipeline 生成缓存"
        )
    gm_path = save_root / dataset_name / "dedup_detections" / "global_mapping.json"
    if not gm_path.exists():
        raise FileNotFoundError(
            f"global_mapping.json 不存在: {gm_path}，请先运行 --mode dedup"
        )

    # 2. 加载 cache
    logger.info("加载 DA3 缓存: %s", cache_path)
    data = np.load(cache_path, allow_pickle=True)
    for k in ("depth", "world_points", "world_points_conf", "extrinsic", "intrinsic"):
        if k not in data:
            raise ValueError(f"DA3 缓存缺少字段: {k}")
    world_points = np.asarray(data["world_points"])          # (N,H,W,3)
    world_points_conf = np.asarray(data["world_points_conf"])  # (N,H,W)
    extrinsic = np.asarray(data["extrinsic"])                # (N,3,4)
    if "image_ids" in data:
        image_ids = [int(x) for x in np.asarray(data["image_ids"]).ravel()]
    else:
        image_ids = list(range(world_points.shape[0]))
    N, H, W, _ = world_points.shape
    logger.info("缓存: N=%d H=%d W=%d image_ids=%s", N, H, W, image_ids)
    if len(image_ids) != N:
        raise ValueError(f"image_ids 数({len(image_ids)})与缓存帧数({N})不一致")

    # 3. 加载 global_mapping -> reverse_mapping + gid_meta
    logger.info("加载 global_mapping: %s", gm_path)
    global_mapping: Dict[str, List[Dict]] = json.loads(gm_path.read_text(encoding="utf-8"))
    reverse_mapping: Dict[Tuple[int, int], int] = {}
    gid_meta: Dict[int, Dict] = {}
    for gid_str, entries in global_mapping.items():
        gid = int(gid_str)
        gid_meta[gid] = {"sku_class": "unknown", "n_instances": len(entries)}
        for e in entries:
            reverse_mapping[(int(e["image_id"]), int(e["object_id"]))] = gid

    # 4. 按帧顺序加载 detection + 构造 transforms（原图 PIL 尺寸 -> cache (W,H)）+ 填 sku_class
    detections: List[Dict] = []
    transforms_info: List[Pi3ImageTransform] = []
    gid_sku_filled = set()
    for img_id in image_ids:
        det_path = detections_dir / f"{img_id}.json"
        if not det_path.exists():
            raise FileNotFoundError(f"检测文件不存在: {det_path}")
        det = json.loads(det_path.read_text(encoding="utf-8"))
        detections.append(det)

        img_path = _resolve_image_path(images_dir, img_id)
        with Image.open(img_path) as im:
            ow, oh = im.size
        transforms_info.append(Pi3ImageTransform(ow, oh, W, H))

        sku_class = _extract_sku_class(det)
        for obj_idx in range(len(_flatten_objects(det))):
            gid = reverse_mapping.get((img_id, obj_idx))
            if gid is not None and gid not in gid_sku_filled:
                gid_meta[gid]["sku_class"] = sku_class
                gid_sku_filled.add(gid)

    # 5. 提取 per-(image,object) 点云并按 global_id 赋值
    logger.info("提取 3D 点云（按 global_id 聚合）...")
    result = extract_3d_from_bboxes(
        world_points=world_points,
        world_points_conf=world_points_conf,
        detections=detections,
        reverse_mapping=reverse_mapping,
        image_ids=image_ids,
        conf_threshold=conf_threshold,
        bbox_margin=bbox_margin,
        return_stats=True,
    )
    points_3d = result["points"]          # (M,3)
    global_ids = result["global_ids"]     # (M,)
    stats = result["stats"]
    logger.info(
        "提取完成: %d 点, %d 唯一 global_id（有效 bbox %d/%d）",
        stats.get("total_points", 0),
        stats.get("unique_global_ids", 0),
        stats.get("valid_bboxes", 0),
        stats.get("total_bboxes", 0),
    )

    # 6. 计算 front-facing 面积
    candidate_normals = camera_view_normals(extrinsic)
    logger.info("候选法向: %d 个相机视线方向", len(candidate_normals))
    report = compute_facing_area_report(
        points_3d=points_3d,
        global_ids=global_ids,
        gid_meta=gid_meta,
        candidate_normals=candidate_normals,
        min_points=min_points,
    )
    logger.info(
        "牌面面积: total=%.4f m², %d/%d global_id 有面积",
        report["total_facing_area_m2"],
        report["n_global_ids_with_area"],
        report["n_global_ids"],
    )

    # 7. 写 facing_area_report.json
    report_path = save_root / dataset_name / "facing_area_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "dataset": dataset_name,
        "backend": backend,
        "n_images": N,
        "n_global_ids": report["n_global_ids"],
        "n_global_ids_with_area": report["n_global_ids_with_area"],
        "total_facing_area_m2": report["total_facing_area_m2"],
        "per_sku_class": report["per_sku_class"],
        "per_global_id": report["per_global_id"],
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("牌面面积报告: %s", report_path)

    # 8. 回写 global_mapping.json：每 entry 追加 facing_area_m2 + facing_share
    per_gid = report["per_global_id"]
    for gid_str, entries in global_mapping.items():
        area = per_gid.get(gid_str, {}).get("area_m2", 0.0)
        share = per_gid.get(gid_str, {}).get("share", 0.0)
        for e in entries:
            e["facing_area_m2"] = area
            e["facing_share"] = share
    gm_path.write_text(json.dumps(global_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已回写 global_mapping.json（追加 facing_area_m2/facing_share）: %s", gm_path)

    return {
        "success": True,
        "report_path": str(report_path),
        "global_mapping_path": str(gm_path),
        "summary": {
            "total_facing_area_m2": report["total_facing_area_m2"],
            "n_global_ids": report["n_global_ids"],
            "n_global_ids_with_area": report["n_global_ids_with_area"],
            "per_sku_class": report["per_sku_class"],
        },
    }
