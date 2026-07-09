from __future__ import annotations

from typing import Dict, Any

from utils.global_id_mapper import GlobalIDMapper


def build_global_object_index(mapper: GlobalIDMapper) -> Dict[str, Any]:
    index: Dict[str, Any] = {}
    for gid in mapper.get_all_global_ids():
        instances = mapper.get_instances(gid)
        active = [inst for inst in instances if not inst.removed]
        index[str(gid)] = {
            "images": mapper.get_images_for_id(gid),
            "objects": mapper.get_object_ids_for_id(gid),
            "active_count": len(active),
            "removed_count": len(instances) - len(active),
            "total_count": len(instances),
            "instances": [inst.to_dict() for inst in instances],
        }
    return index

