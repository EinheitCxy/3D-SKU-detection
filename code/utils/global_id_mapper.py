"""
Global ID Mapper - 全局ID数据管理工具类

用于管理和查询 global_mapping.json 中的全局ID映射数据。
支持从JSON加载、查询全局ID实例、获取图片列表、提取bbox等操作。

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstanceInfo:
    """表示单个物体实例的信息"""

    def __init__(self, image_id: int, object_id: int, bbox: List[float], removed: bool):
        self.image_id = image_id
        self.object_id = object_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.removed = removed

    def __repr__(self) -> str:
        status = "removed" if self.removed else "active"
        return f"Instance(img={self.image_id}, obj={self.object_id}, {status})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "image_id": self.image_id,
            "object_id": self.object_id,
            "bbox": self.bbox,
            "removed": self.removed,
        }


class GlobalIDMapper:
    """
    全局ID映射管理器

    从 global_mapping.json 加载数据，提供便捷的查询接口。

    JSON格式：
    {
      "1": [
        {"image_id": 1, "object_id": 0, "bbox": [x1, y1, x2, y2], "removed": false},
        {"image_id": 2, "object_id": 20, "bbox": [x1, y1, x2, y2], "removed": true},
        ...
      ],
      "2": [...]
    }
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        初始化GlobalIDMapper

        Args:
            json_path: global_mapping.json 文件路径 (可选，可稍后调用 load_from_json)
        """
        self.data: Dict[str, List[InstanceInfo]] = {}
        self.json_path: Optional[Path] = None

        if json_path:
            self.load_from_json(json_path)

    def load_from_json(self, json_path: str) -> None:
        """
        从JSON文件加载全局ID映射数据

        Args:
            json_path: global_mapping.json 文件路径

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Global mapping JSON not found: {json_path}")

        logger.info(f"Loading global ID mapping from: {json_path}")

        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 解析为 InstanceInfo 对象
        self.data = {}
        for global_id_str, instances_list in raw_data.items():
            instances = []
            for inst_dict in instances_list:
                instances.append(
                    InstanceInfo(
                        image_id=inst_dict["image_id"],
                        object_id=inst_dict["object_id"],
                        bbox=inst_dict["bbox"],
                        removed=inst_dict.get("removed", False),
                    )
                )
            self.data[global_id_str] = instances

        self.json_path = path
        logger.info(
            f"Loaded {len(self.data)} global IDs with {self._count_total_instances()} total instances"
        )

    def _count_total_instances(self) -> int:
        """统计总实例数"""
        return sum(len(instances) for instances in self.data.values())

    def get_all_global_ids(self) -> List[int]:
        """
        获取所有全局ID列表（排序后）

        Returns:
            全局ID的整数列表
        """
        return sorted([int(gid) for gid in self.data.keys()])

    def get_instances(self, global_id: int) -> List[InstanceInfo]:
        """
        获取某个全局ID的所有实例（包括已移除的）

        Args:
            global_id: 全局ID

        Returns:
            InstanceInfo 对象列表（可能为空）
        """
        return self.data.get(str(global_id), [])

    def get_active_instances(self, global_id: int) -> List[InstanceInfo]:
        """
        获取某个全局ID的活跃实例（未被去重移除）

        Args:
            global_id: 全局ID

        Returns:
            未被移除的 InstanceInfo 对象列表
        """
        instances = self.get_instances(global_id)
        return [inst for inst in instances if not inst.removed]

    def get_images_for_id(self, global_id: int) -> List[int]:
        """
        获取某个全局ID出现在哪些图片中

        Args:
            global_id: 全局ID

        Returns:
            图片ID列表（排序后）
        """
        instances = self.get_instances(global_id)
        image_ids = list(set(inst.image_id for inst in instances))
        return sorted(image_ids)

    def get_object_ids_for_id(self, global_id: int) -> List[int]:
        """
        获取某个全局ID在各图片中的object_id列表

        Args:
            global_id: 全局ID

        Returns:
            object_id 列表（排序后）
        """
        instances = self.get_instances(global_id)
        object_ids = [inst.object_id for inst in instances]
        return sorted(object_ids)

    def get_bbox_for_instance(
        self, global_id: int, image_id: int
    ) -> Optional[List[float]]:
        """
        获取特定实例的bbox坐标

        Args:
            global_id: 全局ID
            image_id: 图片ID

        Returns:
            bbox 坐标 [x1, y1, x2, y2]，如果不存在返回 None
        """
        instances = self.get_instances(global_id)
        for inst in instances:
            if inst.image_id == image_id:
                return inst.bbox
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Returns:
            包含统计信息的字典：
            - total_global_ids: 全局ID总数
            - total_instances: 实例总数
            - active_instances: 活跃实例数
            - removed_instances: 已移除实例数
            - images_count: 涉及的图片数量
        """
        total_instances = 0
        active_instances = 0
        removed_instances = 0
        all_image_ids = set()

        for instances in self.data.values():
            total_instances += len(instances)
            for inst in instances:
                if inst.removed:
                    removed_instances += 1
                else:
                    active_instances += 1
                all_image_ids.add(inst.image_id)

        return {
            "total_global_ids": len(self.data),
            "total_instances": total_instances,
            "active_instances": active_instances,
            "removed_instances": removed_instances,
            "images_count": len(all_image_ids),
        }

    def find_instances_in_image(self, image_id: int) -> Dict[int, List[InstanceInfo]]:
        """
        查找某张图片中的所有全局ID实例

        Args:
            image_id: 图片ID

        Returns:
            {global_id: [InstanceInfo, ...]} 的映射
        """
        result: Dict[int, List[InstanceInfo]] = {}
        for global_id_str, instances in self.data.items():
            matches = [inst for inst in instances if inst.image_id == image_id]
            if matches:
                result[int(global_id_str)] = matches
        return result

    def get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """
        计算bbox中心点坐标

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def print_summary(self) -> None:
        """打印数据集摘要信息"""
        stats = self.get_statistics()
        print("=" * 60)
        print("Global ID Mapping Summary")
        print("=" * 60)
        print(f"Data Source: {self.json_path}")
        print(f"Total Global IDs: {stats['total_global_ids']}")
        print(f"Total Instances: {stats['total_instances']}")
        print(f"  - Active: {stats['active_instances']}")
        print(f"  - Removed: {stats['removed_instances']}")
        print(f"Images Involved: {stats['images_count']}")
        print("=" * 60)


def _main() -> int:
    """Simple CLI: load a global_mapping.json and print a concise summary."""
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Global ID mapper inspector")
    parser.add_argument("json", type=str, help="Path to global_mapping.json")
    parser.add_argument(
        "--gid",
        type=int,
        default=None,
        help="Optional: print details for a specific Global ID",
    )
    args = parser.parse_args()

    mapper = GlobalIDMapper(args.json)
    mapper.print_summary()

    if args.gid is not None:
        print(f"\nDetails for Global ID = {args.gid}:")
        instances = mapper.get_instances(args.gid)
        print(f"  Total instances: {len(instances)}")
        print(f"  Appears in images: {mapper.get_images_for_id(args.gid)}")
        print(f"  Object IDs: {mapper.get_object_ids_for_id(args.gid)}")
        actives = mapper.get_active_instances(args.gid)
        print(f"  Active instances: {len(actives)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
