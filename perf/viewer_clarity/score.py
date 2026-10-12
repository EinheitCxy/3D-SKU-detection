"""Score source-camera Surfel renders against source images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def score_frame(generation: Path, render_path: Path, frame_index: int, images_dir: Path, scale: int = 2, mode: str = "surfel") -> dict:
    surfel = json.loads((generation / "surfel.json").read_text(encoding="utf-8"))
    frame = surfel["frames"][frame_index]
    width, height = map(int, surfel["grid_size"])
    rendered = cv2.cvtColor(cv2.imread(str(render_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rendered.shape[:2] != (scale * height, scale * width):
        raise ValueError(f"render size mismatch: {rendered.shape[:2]}")
    source = cv2.imread(str(images_dir / f"{int(frame['image_id'])}.jpg"), cv2.IMREAD_COLOR)
    if source is None:
        source = cv2.imread(str(images_dir / f"{int(frame['image_id'])}.JPG"), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"source image id {frame['image_id']} not found")
    expected_size = tuple(map(int, frame["source_size"]))
    if (source.shape[1], source.shape[0]) != expected_size:
        raise ValueError(f"source image size {(source.shape[1], source.shape[0])} differs from surfel metadata {expected_size}")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    affine = np.asarray(frame["processed_to_source"], dtype=np.float64)
    ys, xs = np.indices((scale * height, scale * width), dtype=np.float64)
    px = (xs + 0.5) / scale - 0.5
    py = (ys + 0.5) / scale - 0.5
    sx = affine[0, 0] * px + affine[0, 1] * py + affine[0, 2]
    sy = affine[1, 0] * px + affine[1, 1] * py + affine[1, 2]
    reference = cv2.remap(source, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth_path = generation / "surfel-depth.f16.bin"
    depths = np.fromfile(depth_path, dtype="<f2").reshape(len(surfel["frames"]), height, width)
    valid = cv2.resize((depths[frame_index] > 0).astype(np.uint8), (scale * width, scale * height), interpolation=cv2.INTER_NEAREST).astype(bool)
    refwhite = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) >= 245
    mask = valid & ~refwhite
    hole_color = np.array([255, 0, 255] if mode == "glb" else [255, 255, 255], dtype=np.uint8)
    hole = np.all(rendered == hole_color, axis=2)
    coverage = float((~hole & mask).sum() / mask.sum()) if mask.any() else 0.0
    compared = mask & ~hole
    if compared.any():
        error = rendered.astype(np.float32) - reference.astype(np.float32)
        mse = float(np.mean(error[compared] ** 2))
        psnr = float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
        _, ssim_map = structural_similarity(reference, rendered, channel_axis=-1, data_range=255, full=True)
        per_pixel_ssim = ssim_map.mean(axis=2)
        ssim = float(per_pixel_ssim[compared].mean())
    else:
        psnr, ssim = float("nan"), float("nan")
    return {"frame": frame_index, "coverage": coverage, "PSNR": psnr, "SSIM": ssim, "valid_pixels": int(mask.sum()), "compared_pixels": int(compared.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", required=True, type=Path)
    parser.add_argument("--renders-dir", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int, nargs="+")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--mode", choices=("surfel", "glb"), default="surfel")
    args = parser.parse_args()
    values = [score_frame(args.generation_dir, args.renders_dir / f"frame-{k}.png", k, args.images_dir, args.scale, args.mode) for k in args.frames]
    result = {"frames": values, "mean": {key: float(np.mean([x[key] for x in values])) for key in ("coverage", "PSNR", "SSIM")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
