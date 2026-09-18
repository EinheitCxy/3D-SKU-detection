"""Shared production DA3 preprocessing defaults (no model imports)."""

DEFAULT_PROCESS_RES = 504
PREPROCESS_METHOD = "upper_bound_resize"


def processed_grid(width: int, height: int, process_res: int = DEFAULT_PROCESS_RES) -> tuple[int, int]:
    """DA3's full-image resize followed by nearest-14 resize, without cropping."""
    if min(width, height, process_res) <= 0:
        raise ValueError("图片尺寸和推理分辨率必须大于零")
    scale = process_res / float(max(width, height))
    result = []
    for size in (width, height):
        resized = max(1, int(round(size * scale)))
        down = resized // 14 * 14
        up = down + 14
        result.append(max(1, up if up - resized <= resized - down else down))
    return tuple(result)


def validate_batch_grid(image_sizes, process_res: int = DEFAULT_PROCESS_RES) -> tuple[int, int]:
    grids = {processed_grid(int(w), int(h), process_res) for w, h in image_sizes}
    if len(grids) != 1:
        raise ValueError("同一任务的图片缩放后网格不一致，不允许自动裁切边缘；请按相同比例和方向分别提交")
    return next(iter(grids))
