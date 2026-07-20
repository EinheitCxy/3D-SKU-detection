"""
SKU 检测 CLI：YOLO 检测图片，输出 detections_results/*.json。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = Path(__file__).resolve().parent / "checkpoints" / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description="SKU 检测：输出 detections_results/*.json")
    ap.add_argument("input", help="图片目录或单张图片路径")
    ap.add_argument(
        "-o", "--output", default="output", help="输出目录（默认 ./output）"
    )
    ap.add_argument("-m", "--model", default=str(DEFAULT_MODEL), help="YOLO 权重路径")
    ap.add_argument("--conf", type=float, default=0.35, help="置信度阈值（默认 0.35）")
    ap.add_argument("--iou", type=float, default=0.25, help="NMS IoU 阈值（默认 0.25）")
    ap.add_argument("--device", default="cpu", help="推理设备 cuda:0 / cpu（默认 cpu）")
    a = ap.parse_args()

    src = Path(a.input).resolve()
    if src.is_file():
        imgs = [src]
    else:
        imgs = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"未找到图片：{src}")
    if bad := [f.name for f in imgs if not f.stem.isdigit()]:
        raise SystemExit(f"非数字文件名（下游 int(stem) 会丢弃）：{bad}")

    json_dir = Path(a.output).resolve() / "detections_results"
    json_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)

    total = 0
    for img in imgs:
        try:
            r = model.predict(
                str(img),
                conf=a.conf,
                iou=a.iou,
                device=a.device,
                verbose=False,
                save=False,
            )[0]
            objs, names = [], []
            if len(r.boxes):
                b = r.boxes
                for box, c, cf in zip(
                    b.xyxy.cpu().numpy().round().astype(int),
                    b.cls.cpu().numpy().astype(int),
                    b.conf.cpu().numpy(),
                ):
                    objs.append(
                        {
                            "position": box.tolist(),
                            "classes": {"det": int(c)},
                            "confidences": {"det": float(cf)},
                        }
                    )
                    if (n := model.names.get(int(c), str(int(c)))) not in names:
                        names.append(n)
            sku = {"skus": [{"classes": {"det": names}, "objects": objs}]}
        except Exception as e:
            sku, objs = {"skus": [{"classes": {"det": []}, "objects": []}]}, []
            print(f"  ⚠️ {img.name}: 检测失败，写空占位：{e}")
        (json_dir / f"{img.stem}.json").write_text(
            json.dumps(sku, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total += len(objs)
        print(f"  {img.name}: {len(objs)} 个框")
    print(f"完成：共 {total} 个检测框，输出 {json_dir}")


if __name__ == "__main__":
    main()
