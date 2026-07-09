"""SKU 检测 CLI：扫描图片目录/单张图片，用 YOLO 检测，输出可视化图 + 下游兼容 JSON。

输出 JSON 格式与 code/Global-ID-Mapping 的 detections_results 对齐：
    {"skus": [{"classes": {"det": ["<name>"]},
              "objects": [{"position": [x1,y1,x2,y2],   # 整数像素
                           "classes": {"det": <cls_idx>},
                           "confidences": {"det": <conf>}}]}]}

下游 code/utils/data_utils.py::load_detections 用 int(stem) 解析文件名，
因此源图片文件名必须为纯整数（1.jpg, 2.jpg, ...）；非数字名会被本脚本拒绝。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = Path(__file__).resolve().parent / "checkpoints" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SKU 检测：扫描图片目录或单张图片，输出可视化图 + 下游兼容 JSON。"
    )
    p.add_argument("input", help="图片目录或单张图片路径（.jpg/.jpeg/.png）")
    p.add_argument("-o", "--output", default="output", help="输出目录（默认 ./output）")
    p.add_argument(
        "-m", "--model", default=str(DEFAULT_MODEL), help=f"YOLO 权重路径（默认 {DEFAULT_MODEL}）"
    )
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（默认 640，须为 32 的倍数）")
    p.add_argument("--conf", type=float, default=0.35, help="置信度阈值，范围 (0,1]（默认 0.35）")
    p.add_argument("--iou", type=float, default=0.25, help="NMS IoU 阈值，范围 (0,1]（默认 0.25）")
    p.add_argument("--batch", type=int, default=1, help="批处理大小（默认 1=逐图；>1 启用批量预测）")
    p.add_argument("--device", default=None, help="推理设备，如 cuda:0 / cpu（默认自动）")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0 < args.conf <= 1):
        raise SystemExit(f"--conf 必须在 (0, 1] 范围内，当前 {args.conf}")
    if not (0 < args.iou <= 1):
        raise SystemExit(f"--iou 必须在 (0, 1] 范围内，当前 {args.iou}")
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        raise SystemExit(f"--imgsz 必须为正且为 32 的倍数（ultralytics 约束），当前 {args.imgsz}")
    if args.batch < 1:
        raise SystemExit(f"--batch 必须 >= 1，当前 {args.batch}")


def _is_int_stem(stem: str) -> bool:
    try:
        int(stem)
        return True
    except ValueError:
        return False


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMG_EXTS:
            raise SystemExit(f"不支持的图片格式：{input_path.suffix}")
        imgs = [input_path]
    elif input_path.is_dir():
        imgs = sorted(
            f for f in input_path.rglob("*") if f.suffix.lower() in IMG_EXTS and f.is_file()
        )
        if not imgs:
            raise SystemExit(f"目录中未找到图片：{input_path}")
    else:
        raise SystemExit(f"输入路径不存在：{input_path}")

    bad = [img.name for img in imgs if not _is_int_stem(img.stem)]
    if bad:
        raise SystemExit(
            "检测到非数字文件名，已中止：\n  " + ", ".join(bad)
            + "\n下游 code/utils/data_utils.py 用 int(stem) 解析 detections_results 文件名，"
            "非数字名会被静默丢弃。请将图片重命名为纯整数（1.jpg, 2.jpg, ...）后再运行。"
        )
    return imgs


def build_sku_result(result, class_names: dict) -> dict:
    """将单张图的 YOLO 结果转为下游兼容的 {skus:[...]} 结构。"""
    objects: List[dict] = []
    seen_names: List[str] = []
    if len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy().round().astype(int).tolist()
        clses = result.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = result.boxes.conf.cpu().numpy().tolist()
        for box, cls_idx, conf in zip(boxes, clses, confs):
            name = class_names.get(cls_idx, str(cls_idx))
            objects.append(
                {
                    "position": box,  # [x1, y1, x2, y2] 整数像素
                    "classes": {"det": cls_idx},
                    "confidences": {"det": conf},
                }
            )
            if name not in seen_names:
                seen_names.append(name)
    return {"skus": [{"classes": {"det": seen_names}, "objects": objects}]}


def _empty_sku_result() -> dict:
    return {"skus": [{"classes": {"det": []}, "objects": []}]}


def _save_one(result, img_path: Path, class_names: dict, images_dir: Path, json_dir: Path) -> int:
    """保存单张结果：可视化图 + JSON，返回检测框数。"""
    stem = img_path.stem
    sku_result = build_sku_result(result, class_names)
    result.save(str(images_dir / f"{stem}.jpg"))
    (json_dir / f"{stem}.json").write_text(
        json.dumps(sku_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(sku_result["skus"][0]["objects"])


def _write_placeholder(stem: str, json_dir: Path) -> None:
    """检测失败的图写空占位 JSON，保持 detections_results 与源图对齐。"""
    (json_dir / f"{stem}.json").write_text(
        json.dumps(_empty_sku_result(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _predict_one(model, img_path: Path, args, class_names, images_dir, json_dir) -> tuple[int, bool]:
    """逐图预测+保存。失败写空占位并告警，不中断后续图、不破坏对齐。"""
    try:
        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=False,
        )
        n = _save_one(results[0], img_path, class_names, images_dir, json_dir)
        return n, True
    except Exception as e:
        _write_placeholder(img_path.stem, json_dir)
        print(f"  ⚠️ {img_path.name}: 检测失败，已写空占位并跳过（不破坏其余对齐）：{e}")
        return 0, False


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    images_dir = output_dir / "images"
    json_dir = output_dir / "detections_results"
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    model = YOLO(args.model)
    class_names = model.names

    print(f"输入：{input_path}（{len(images)} 张图片）")
    print(f"模型：{args.model}（{len(class_names)} 类）")
    print(f"输出：{output_dir}（batch={args.batch}）")

    total = 0
    failed = 0
    if args.batch > 1:
        for i in range(0, len(images), args.batch):
            chunk = images[i : i + args.batch]
            try:
                results = model.predict(
                    [str(p) for p in chunk],
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                    save=False,
                    batch=args.batch,
                )
            except Exception as e:
                for p in chunk:
                    _write_placeholder(p.stem, json_dir)
                failed += len(chunk)
                print(f"  ⚠️ 批次 [{i},{i + len(chunk)}) 检测失败，已写空占位并跳过：{e}")
                continue
            if len(results) != len(chunk):
                print(f"  ⚠️ 批次 [{i},{i + len(chunk)}) 结果数与输入不符，回退逐图")
                for p in chunk:
                    n, ok = _predict_one(model, p, args, class_names, images_dir, json_dir)
                    if ok:
                        total += n
                        print(f"  {p.name}: {n} 个框")
                    else:
                        failed += 1
                continue
            for p, result in zip(chunk, results):
                try:
                    n = _save_one(result, p, class_names, images_dir, json_dir)
                    total += n
                    print(f"  {p.name}: {n} 个框")
                except Exception as e:
                    _write_placeholder(p.stem, json_dir)
                    failed += 1
                    print(f"  ⚠️ {p.name}: 保存失败，已写空占位并跳过：{e}")
    else:
        for img_path in images:
            n, ok = _predict_one(model, img_path, args, class_names, images_dir, json_dir)
            if ok:
                total += n
                print(f"  {img_path.name}: {n} 个框")
            else:
                failed += 1

    print(f"完成：共 {total} 个检测框，失败 {failed} 张，输出目录 {output_dir}")


if __name__ == "__main__":
    main()
