import torch
import numpy as np
import cv2
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext

# 导入 VGGT 相关模块
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sku_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置类
@dataclass
class SKUMatchingConfig:
    """SKU匹配配置参数"""
    max_points_per_bbox: int = 50  # 每个检测框最大采样点数
    visibility_threshold: float = 0.8
    min_visible_points: int = 8
    max_bboxes: int = 50
    device: str = "cuda"
    dtype: torch.dtype = None
    output_dir: str = "output"
    det_conf_threshold: float = 0.0  # 检测置信度阈值
    min_bbox_area: float = 25.0      # 忽略极小框
    max_total_points: int = 5000     # 全局最大采样点数上限（控制内存/速度）
    seed: Optional[int] = 42         # 全流程随机种子（可复现实验）
    use_autocast: bool = True        # 仅在CUDA上启用autocast
    save_json: bool = False          # 是否将结果保存为JSON
    json_filename: str = "correspondences.json"  # 结果保存文件名
    
    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

# --- Helper Functions ---

def load_detections(detection_file: str) -> List[Dict]:
    """加载检测结果文件
    
    Args:
        detection_file: 检测结果JSON文件路径
        
    Returns:
        检测结果列表
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不正确时抛出
    """
    detection_path = Path(detection_file)
    if not detection_path.exists():
        raise FileNotFoundError(f"Detection file not found: {detection_file}")
    
    try:
        with open(detection_path, 'r', encoding='utf-8') as f:
            detections = json.load(f)
        
        logger.info(f"Loaded {len(detections)} detection results from {detection_file}")
        return detections
        
    except Exception as e:
        logger.error(f"Failed to load detections from {detection_file}: {e}")
        raise

def extract_bboxes_from_detections(detections: List[Dict], image_idx: int, config: SKUMatchingConfig) -> List[Dict]:
    """从检测结果中提取边界框
    
    Args:
        detections: 检测结果列表
        image_idx: 图像索引
        config: 配置参数
        
    Returns:
        边界框列表
    """
    if image_idx >= len(detections):
        logger.warning(f"Image index {image_idx} out of range for {len(detections)} detections")
        return []
    
    detection_data = detections[image_idx]
    if 'objects' not in detection_data:
        logger.warning(f"No objects found in detection data for image {image_idx}")
        return []
    
    bboxes = []
    for obj_idx, obj in enumerate(detection_data['objects']):
        if 'position' in obj:
            x1, y1, x2, y2 = obj['position']
            confidence = obj.get('confidences', {}).get('det', 0.0)
            if confidence < config.det_conf_threshold:
                continue
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area < config.min_bbox_area:
                continue
            bbox_info = {
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'confidence': confidence,
                'object_id': obj_idx,
                'area': area
            }
            bboxes.append(bbox_info)
    
    # 按面积排序并限制数量
    bboxes.sort(key=lambda x: x['area'], reverse=True)
    if len(bboxes) > config.max_bboxes:
        logger.warning(f"Limiting to top {config.max_bboxes} bboxes (found {len(bboxes)})")
        bboxes = bboxes[:config.max_bboxes]
    
    logger.info(f"Extracted {len(bboxes)} bounding boxes from image {image_idx}")
    return bboxes

class VGGTImageTransform:
    """封装与 VGGT load_and_preprocess_images(crop) 一致的几何映射，支持点/框双向映射。"""

    def __init__(self, orig_width: int, orig_height: int, target_size: int = 518):
        self.orig_width = int(orig_width)
        self.orig_height = int(orig_height)
        # crop 模式：固定宽到 target_size，高按比例并取整到 14 的倍数
        self.proc_width = int(target_size)
        self.proc_height = int(round(self.orig_height * (self.proc_width / self.orig_width) / 14) * 14)

        self.scale_x = self.proc_width / self.orig_width
        self.scale_y = self.proc_height / self.orig_height

        # 如果高度超出 target，做中心裁剪；offset_y 为负，表示坐标系向上平移
        if self.proc_height > target_size:
            crop_start_y = (self.proc_height - target_size) // 2
            self.offset_y = -float(crop_start_y)
            self.final_height = int(target_size)
        else:
            self.offset_y = 0.0
            self.final_height = int(self.proc_height)

        self.offset_x = 0.0
        self.final_width = int(self.proc_width)

    def apply_batch_padding(self, max_width: int, max_height: int) -> None:
        """对齐到批次中最大尺寸，居中 pad，更新 offset 与最终尺寸。"""
        if self.final_width < max_width:
            self.offset_x += (max_width - self.final_width) // 2
            self.final_width = int(max_width)
        if self.final_height < max_height:
            self.offset_y += (max_height - self.final_height) // 2
            self.final_height = int(max_height)

    # -------- 原图 -> 模型输入(final) 映射 --------
    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        xp = x * self.scale_x + self.offset_x
        yp = y * self.scale_y + self.offset_y
        xp = max(0.0, min(xp, self.final_width - 1))
        yp = max(0.0, min(yp, self.final_height - 1))
        return xp, yp

    def map_points_to_final(self, points):
        """points: 形如 (..., 2) 的 numpy 数组或 torch 张量。返回同类型同形状。"""
        is_torch = torch.is_tensor(points)
        if is_torch:
            xp = points[..., 0] * self.scale_x + self.offset_x
            yp = points[..., 1] * self.scale_y + self.offset_y
            xp = xp.clamp(0, self.final_width - 1)
            yp = yp.clamp(0, self.final_height - 1)
            out = torch.stack([xp, yp], dim=-1)
        else:
            xp = points[..., 0] * self.scale_x + self.offset_x
            yp = points[..., 1] * self.scale_y + self.offset_y
            xp = np.clip(xp, 0, self.final_width - 1)
            yp = np.clip(yp, 0, self.final_height - 1)
            out = np.stack([xp, yp], axis=-1)
        return out

    def map_bbox_to_final(self, bbox: List[float]) -> List[float]:
        x1p, y1p = self.map_xy_to_final(bbox[0], bbox[1])
        x2p, y2p = self.map_xy_to_final(bbox[2], bbox[3])
        return [x1p, y1p, x2p, y2p]

    # -------- 模型输入(final) -> 原图 映射 --------
    def map_xy_to_original(self, xp: float, yp: float) -> Tuple[float, float]:
        sx = self.scale_x if self.scale_x != 0 else 1.0
        sy = self.scale_y if self.scale_y != 0 else 1.0
        x = (xp - self.offset_x) / sx
        y = (yp - self.offset_y) / sy
        x = max(0.0, min(x, self.orig_width - 1))
        y = max(0.0, min(y, self.orig_height - 1))
        return x, y

    def map_points_to_original(self, points):
        is_torch = torch.is_tensor(points)
        if is_torch:
            sx = self.scale_x if self.scale_x != 0 else 1.0
            sy = self.scale_y if self.scale_y != 0 else 1.0
            x = (points[..., 0] - self.offset_x) / sx
            y = (points[..., 1] - self.offset_y) / sy
            x = x.clamp(0, self.orig_width - 1)
            y = y.clamp(0, self.orig_height - 1)
            out = torch.stack([x, y], dim=-1)
        else:
            sx = self.scale_x if self.scale_x != 0 else 1.0
            sy = self.scale_y if self.scale_y != 0 else 1.0
            x = (points[..., 0] - self.offset_x) / sx
            y = (points[..., 1] - self.offset_y) / sy
            x = np.clip(x, 0, self.orig_width - 1)
            y = np.clip(y, 0, self.orig_height - 1)
            out = np.stack([x, y], axis=-1)
        return out

    def map_bbox_to_original(self, bbox: List[float]) -> List[float]:
        x1, y1 = self.map_xy_to_original(bbox[0], bbox[1])
        x2, y2 = self.map_xy_to_original(bbox[2], bbox[3])
        return [x1, y1, x2, y2]


def build_vggt_transforms(image_paths: List[str], target_size: int = 518) -> List[VGGTImageTransform]:
    """构建与 load_and_preprocess_images(crop) 一致的每张图像的几何映射，并匹配批次 pad。"""
    from PIL import Image as _Image

    transforms: List[VGGTImageTransform] = []
    for p in image_paths:
        img = _Image.open(p).convert("RGB")
        w, h = img.size
        transforms.append(VGGTImageTransform(w, h, target_size=target_size))

    # 批次对齐 pad
    max_w = max(t.final_width for t in transforms)
    max_h = max(t.final_height for t in transforms)
    for t in transforms:
        t.apply_batch_padding(max_w, max_h)

    return transforms

def generate_points_from_bboxes(bboxes: List[Dict], image_shape: Tuple[int, int], config: SKUMatchingConfig) -> Tuple[Optional[torch.Tensor], Dict[int, Dict]]:
    """在检测框内随机采样点作为查询点
    
    Args:
        bboxes: 检测框列表
        image_shape: 图像形状 (height, width)
        config: 配置参数
        
    Returns:
        tuple: (查询点张量, 物体点映射字典)
    """
    logger.info(f"Generating query points from {len(bboxes)} bounding boxes...")
    
    all_query_points = []
    points_per_object = {}
    
    height, width = image_shape
    total_points = 0
    
    try:
        for bbox_info in bboxes:
            x1, y1, x2, y2 = bbox_info['bbox']
            object_id = bbox_info['object_id']
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # 在检测框内生成网格点
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # 计算需要的网格密度
            area = bbox_width * bbox_height
            if area == 0:
                logger.warning(f"Invalid bbox area for object {object_id}, skipping")
                continue
            
            # 基于面积计算采样点数，但不超过最大值
            desired_points = min(int(np.sqrt(area) * 2), config.max_points_per_bbox)
            if desired_points < 5:  # 至少5个点
                desired_points = 5
            
            # 计算网格大小
            grid_cols = max(1, int(np.sqrt(desired_points * bbox_width / bbox_height)))
            grid_rows = max(1, int(np.sqrt(desired_points * bbox_height / bbox_width)))
            
            # 生成网格点
            x_points = np.linspace(x1, x2, grid_cols)
            y_points = np.linspace(y1, y2, grid_rows)
            
            # 创建网格
            xx, yy = np.meshgrid(x_points, y_points)
            xx = xx.flatten()
            yy = yy.flatten()
            
            # 随机选择点以达到 desired_points 数量
            if len(xx) > desired_points:
                indices = np.random.choice(len(xx), desired_points, replace=False)
                xx = xx[indices]
                yy = yy[indices]
            
            # 使用numpy操作提高性能
            points = np.stack([xx, yy], axis=-1).astype(np.float32)

            # 全局点数上限控制
            remaining_allowance = config.max_total_points - total_points
            if remaining_allowance <= 0:
                logger.warning("Reached max_total_points limit; stopping further point sampling.")
                break
            if len(points) > remaining_allowance:
                sel_idx = np.random.choice(len(points), remaining_allowance, replace=False)
                points = points[sel_idx]

            all_query_points.append(points)

            num_points = len(points)
            start_idx = total_points
            points_per_object[object_id] = {
                "point_indices": (start_idx, start_idx + num_points),
                "bbox": bbox_info['bbox'],
                "center": bbox_info['center'],
                "confidence": bbox_info['confidence'],
                "area": bbox_info['area'],
                "num_sampled_points": num_points,
                "original_bbox": bbox_info.get('original_bbox', bbox_info['bbox'])
            }
            total_points += num_points
            
        if not all_query_points:
            logger.warning("No query points generated from bounding boxes")
            return None, None
            
        # 一次性转换为torch tensor，减少内存碎片
        all_points_array = np.concatenate(all_query_points, axis=0)
        query_points_tensor = torch.from_numpy(all_points_array).float()
        
        logger.info(f"Generated {total_points} query points from {len(points_per_object)} objects")
        return query_points_tensor, points_per_object
        
    except Exception as e:
        logger.error(f"Failed to generate points from bounding boxes: {e}")
        raise

def visualize_results(images: torch.Tensor, reference_idx: int, points_per_object: Dict[int, Dict], 
                     correspondences: Dict[int, List[Dict]], config: SKUMatchingConfig) -> None:
    """可视化追踪结果
    
    Args:
        images: 图像张量 (S, C, H, W)
        reference_idx: 参考图像索引
        points_per_object: 物体点映射
        correspondences: 对应关系结果
        config: 配置参数
    """
    logger.info("Generating visualization results...")
    
    try:
        # 1. 可视化参考图像和其上的检测框
        ref_image_np = (images[reference_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ref_image_bgr = cv2.cvtColor(ref_image_np, cv2.COLOR_RGB2BGR)
        
        overlay = ref_image_bgr.copy()
        colors = {}
        
        # 绘制参考图像的检测框和采样点
        for obj_id, data in points_per_object.items():
            bbox = data['bbox']
            center = data['center']
            
            # 生成稳定的颜色
            colors[obj_id] = np.random.randint(0, 255, (3,)).tolist()
            color = colors[obj_id]
            
            # 绘制检测框
            cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # 绘制中心点
            cv2.circle(overlay, (int(center[0]), int(center[1])), 5, color, -1)
            
            # 绘制ID标签
            cv2.putText(overlay, f"ID: {obj_id}", (int(bbox[0]), int(bbox[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        ref_output_path = Path(config.output_dir) / "reference_image_with_bboxes.jpg"
        cv2.imwrite(str(ref_output_path), overlay)
        logger.info(f"Saved reference image with bounding boxes to '{ref_output_path}'")

        # 2. 可视化每个目标图像上的对应边界框
        for s_idx, new_boxes in correspondences.items():
            target_image_np = (images[s_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            target_image_bgr = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2BGR)
            
            for item in new_boxes:
                obj_id = item['object_id']
                box = [int(c) for c in item['box']]
                confidence = item.get('confidence', 0.0)
                
                # 使用预定义的颜色
                color = colors.get(obj_id, [255, 255, 255])

                # 绘制边界框
                cv2.rectangle(target_image_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # 绘制ID和置信度
                label = f"ID: {obj_id} ({confidence:.2f})"
                cv2.putText(target_image_bgr, label, (box[0], box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            output_filename = Path(config.output_dir) / f"target_image_{s_idx}_correspondences.jpg"
            cv2.imwrite(str(output_filename), target_image_bgr)
            logger.info(f"Saved target image correspondences to '{output_filename}'")
            
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        raise


def save_correspondences_json(
    correspondences: Dict[int, List[Dict]],
    points_per_object: Optional[Dict[int, Dict]],
    config: SKUMatchingConfig,
    meta: Optional[Dict] = None,
) -> Path:
    """将匹配结果保存为 JSON 文件
    
    Args:
        correspondences: 匹配结果
        points_per_object: 参考图像对象点信息
        config: 配置
        meta: 可选元数据（如图像路径列表、时间戳等）
    Returns:
        保存文件路径
    """
    try:
        result = {
            "correspondences": correspondences,
            "reference_points": points_per_object if points_per_object is not None else {},
            "meta": meta or {},
        }
        out_path = Path(config.output_dir) / config.json_filename
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved correspondences JSON to '{out_path}'")
        return out_path
    except Exception as e:
        logger.error(f"Failed to save correspondences JSON: {e}")
        raise

# --- Main Function ---

def find_object_correspondences(
    vggt_model: VGGT,
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[VGGTImageTransform]] = None,
) -> Tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
    """查找物体对应关系的主函数
    
    Args:
        vggt_model: VGGT模型
        detections: 检测结果列表
        images: 图像张量 (S, C, H, W)
        config: 配置参数
        reference_image_idx: 参考图像索引
        
    Returns:
        tuple: (对应关系结果, 物体点映射)
    """
    logger.info("Starting object correspondence detection...")
    
    try:
        S = images.shape[0]
        _, _, H, W = images.shape
        device = images.device
        
        # 验证输入参数
        if reference_image_idx >= S:
            raise ValueError(f"Reference image index {reference_image_idx} out of range for {S} images")
        
        # 1. 从检测结果中提取参考图像的边界框
        logger.info(f"Processing reference image {reference_image_idx}")
        ref_bboxes = extract_bboxes_from_detections(detections, reference_image_idx, config)

        if not ref_bboxes:
            logger.warning(f"No bounding boxes found in reference image {reference_image_idx}")
            return {}, None

        # 2. 原图尺寸：优先使用 transforms_info（VGGTImageTransform），否则回退到 VGGT 输入尺寸
        if transforms_info is None or not (0 <= reference_image_idx < len(transforms_info)):
            logger.warning("transforms_info missing; falling back to VGGT input size for original size.")
            raise ValueError("transforms_info missing")
        else:
            ref_transform = transforms_info[reference_image_idx]
            orig_h = int(ref_transform.orig_height)
            orig_w = int(ref_transform.orig_width)
        logger.info(f"Original image size: {orig_w}x{orig_h}, VGGT input size: {W}x{H}")

        # 3. 为每个边界框生成查询点（严格按 load_and_preprocess_images 的几何映射对齐）
        
        # 标准路径：使用 transforms_info（源自 PIL.Image.size）进行严格对齐
        mapped_bboxes = []
        for b in ref_bboxes:
            mapped = ref_transform.map_bbox_to_final(b['bbox'])
            b2 = dict(b)
            b2['original_bbox'] = b['bbox']
            b2['bbox'] = mapped
            b2['center'] = [(mapped[0] + mapped[2]) / 2, (mapped[1] + mapped[3]) / 2]
            b2['area'] = max(0.0, (mapped[2] - mapped[0]) * (mapped[3] - mapped[1]))
            mapped_bboxes.append(b2)
        ref_bboxes = mapped_bboxes
        logger.info("Mapped detection bboxes to preprocessed input coordinates via transforms_info.")

        all_query_points_tensor, points_per_object = generate_points_from_bboxes(
            ref_bboxes, (H, W), config
        )
        
        if all_query_points_tensor is None:
            logger.warning("Could not generate query points from bounding boxes.")
            return {}, None

        logger.info(f"Generated {len(all_query_points_tensor)} total query points.")
        all_query_points_tensor = all_query_points_tensor.to(device)

        # 4. 使用 VGGT 执行点追踪
        logger.info("Tracking points with VGGT...")
        start_time = time.time()
        
        with torch.no_grad():
            try:
                predictions = vggt_model(images.unsqueeze(0), query_points=all_query_points_tensor.unsqueeze(0))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                    logger.error("CUDA out of memory during tracking. Trying to free cache and fail fast.")
                    torch.cuda.empty_cache()
                raise
        
        tracks = predictions['track'].squeeze(0)      # 点轨迹 (S, N, 2)
        visibility = predictions['vis'].squeeze(0)    # 可见性分数 (S, N)
        tracking_time = time.time() - start_time
        logger.info(f"Tracking complete in {tracking_time:.2f}s")

        # 5. 在每个目标图像中重建物体边界框
        logger.info("Reconstructing object bounding boxes...")
        object_correspondences = {}
        
        for s_idx in range(S):
            if s_idx == reference_image_idx:
                continue

            new_boxes_in_image = []
            for object_id, data in points_per_object.items():
                start, end = data["point_indices"]
                object_tracks = tracks[s_idx, start:end, :]
                object_visibility = visibility[s_idx, start:end]

                visible_mask = object_visibility > config.visibility_threshold
                visible_points = object_tracks[visible_mask]

                # 过滤非有限值，并裁剪到图像范围
                if visible_points.numel() > 0:
                    finite_mask = torch.isfinite(visible_points).all(dim=1)
                    visible_points = visible_points[finite_mask]
                if visible_points.numel() == 0:
                    continue
                # clamp 到 [0, W-1] 与 [0, H-1]
                visible_points[:, 0] = visible_points[:, 0].clamp_(0, W - 1)
                visible_points[:, 1] = visible_points[:, 1].clamp_(0, H - 1)

                if len(visible_points) < config.min_visible_points:
                    continue

                x_min, y_min = visible_points.min(dim=0).values
                x_max, y_max = visible_points.max(dim=0).values
                
                # 将模型输入坐标回映射到原图坐标
                vggt_box = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                if transforms_info is not None and 0 <= s_idx < len(transforms_info):
                    scaled_box = transforms_info[s_idx].map_bbox_to_original(vggt_box)
                else:
                    raise ValueError("transforms_info missing or index is invalid")
                
                new_boxes_in_image.append({
                    'object_id': object_id,
                    'box': scaled_box,
                    'vggt_box': [x_min.item(), y_min.item(), x_max.item(), y_max.item()],
                    'confidence': float(object_visibility.mean().item()),
                    'num_points': len(visible_points),
                    'original_confidence': data['confidence'],
                    'original_bbox': data['bbox']
                })
            
            if new_boxes_in_image:
                object_correspondences[s_idx] = new_boxes_in_image
                logger.info(f"Found {len(new_boxes_in_image)} objects in image {s_idx}")

        logger.info(f"Object correspondence detection complete. Found correspondences in {len(object_correspondences)} images.")
        return object_correspondences, points_per_object
        
    except Exception as e:
        logger.error(f"Failed to find object correspondences: {e}")
        raise


# --- Execution Example ---

class SKUMatchingSystem:
    """SKU匹配系统类"""
    
    def __init__(self, config: SKUMatchingConfig = None):
        """初始化SKU匹配系统
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config or SKUMatchingConfig()
        self.vggt_model = None
        self._is_initialized = False
        
    def initialize(self) -> None:
        """初始化模型"""
        if self._is_initialized:
            logger.info("Models already initialized")
            return
            
        logger.info("Initializing SKU matching system...")
        
        try:
            # 设置随机种子（可复现）
            if self.config.seed is not None:
                try:
                    import random
                    random.seed(self.config.seed)
                except Exception:
                    pass
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)

            # 加载 VGGT 模型
            logger.info("Loading VGGT model...")
            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.config.device).eval()
            logger.info("VGGT model loaded successfully")
            
            self._is_initialized = True
            logger.info("SKU matching system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def process_images(self, image_folder: str, detection_file: str, 
                      reference_image_idx: int = 0, max_images: int = None) -> Dict[int, List[Dict]]:
        """处理图像文件夹
        
        Args:
            image_folder: 图像文件夹路径
            detection_file: 检测结果文件路径
            reference_image_idx: 参考图像索引
            max_images: 最大处理图像数量
            
        Returns:
            对应关系结果
        """
        if not self._is_initialized:
            self.initialize()
            
        try:
            # 加载检测结果
            logger.info("Loading detection results...")
            detections = load_detections(detection_file)
            
            # 加载和预处理图像
            image_folder_path = Path(image_folder)
            if not image_folder_path.exists():
                raise FileNotFoundError(f"Image folder not found: {image_folder}")
                
            image_paths = sorted([
                str(image_folder_path / f) 
                for f in os.listdir(image_folder) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            
            if not image_paths:
                raise ValueError(f"No images found in {image_folder}")
            
            if max_images is not None:
                image_paths = image_paths[:max_images]
            
            logger.info(f"Loading {len(image_paths)} images from {image_folder}")
            # 计算与 VGGT 相同的几何预处理映射（用于 bbox/点 坐标对齐），再实际加载图像
            transforms_info = build_vggt_transforms(image_paths, target_size=518)
            images = load_and_preprocess_images(image_paths, mode="crop").to(self.config.device)
            
            # 运行物体对应流程
            logger.info("Running object correspondence detection...")
            use_amp = (
                self.config.use_autocast
                and torch.cuda.is_available()
                and isinstance(self.config.dtype, torch.dtype)
                and (isinstance(self.config.device, str) and self.config.device.startswith("cuda"))
            )
            amp_ctx = torch.cuda.amp.autocast(dtype=self.config.dtype) if use_amp else nullcontext()
            with amp_ctx:
                correspondences, points_map = find_object_correspondences(
                    self.vggt_model,
                    detections,
                    images,
                    self.config,
                    reference_image_idx=reference_image_idx,
                    transforms_info=transforms_info
                )
            
            # 可视化结果
            if correspondences:
                logger.info("Generating visualization...")
                visualize_results(images, reference_image_idx, points_map, correspondences, self.config)
                
                # 打印结果摘要
                self._print_results_summary(correspondences)

                # 可选保存 JSON
                if self.config.save_json:
                    meta = {
                        "image_paths": image_paths,
                        "reference_image_idx": reference_image_idx,
                        "config": {
                            "visibility_threshold": self.config.visibility_threshold,
                            "min_visible_points": self.config.min_visible_points,
                            "max_points_per_bbox": self.config.max_points_per_bbox,
                            "max_bboxes": self.config.max_bboxes,
                        },
                    }
                    save_correspondences_json(correspondences, points_map, self.config, meta)
            else:
                logger.warning("No object correspondences found")
            
            # 可选：返回时附带 transforms_info 以便上游复用（不改变现有返回结构）
            return correspondences
            
        except Exception as e:
            logger.error(f"Failed to process images: {e}")
            raise
    
    def _print_results_summary(self, correspondences: Dict[int, List[Dict]]) -> None:
        """打印结果摘要"""
        logger.info("\n=== Object Correspondences Summary ===")
        for target_idx, found_objects in correspondences.items():
            logger.info(f"\nTarget Image {target_idx}:")
            for obj in found_objects:
                logger.info(f"  - Object ID {obj['object_id']}: box={[round(c, 2) for c in obj['box']]}, "
                           f"confidence={obj['confidence']:.3f}, points={obj['num_points']}")


if __name__ == '__main__':
    # 示例使用
    try:
        # 创建配置
        config = SKUMatchingConfig(
            max_points_per_bbox=30,
            visibility_threshold=0.7,
            min_visible_points=6,
            output_dir="output_results"
        )
        
        # 创建系统实例
        sku_system = SKUMatchingSystem(config)
        
        # 处理图像
        image_folder = "../imdata"
        detection_file = "../sku_detection.json"
        correspondences = sku_system.process_images(
            image_folder=image_folder,
            detection_file=detection_file,
            reference_image_idx=0,
            max_images=5
        )
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
