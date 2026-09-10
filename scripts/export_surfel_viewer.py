"""Export an independent schema-3 bundle with depth-constrained textured surfels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.web_viewer_export import export_web_viewer_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-cache-root", type=Path, required=True)
    parser.add_argument("--sku-masterdata", type=Path, default=Path("runtime/sku_masterdata.csv"))
    parser.add_argument("--texture-edge", type=int, default=1920,
                        help="Texture longest edge in [256, 1920]; short edge capped at 1080, no upscaling")
    parser.add_argument("--voxel-size", type=float, default=0.005)
    args = parser.parse_args()
    print(json.dumps(export_web_viewer_bundle(
        dataset_name=args.dataset.name, da3_cache_path=args.cache,
        global_mapping_path=args.mapping, output_dir=args.output,
        source_images_dir=args.dataset / "images", sam3_mask_cache_root=args.mask_cache_root,
        sku_masterdata_csv=args.sku_masterdata, surfel_texture_edge=args.texture_edge,
        voxel_size_m=args.voxel_size,
    ), indent=2))


if __name__ == "__main__":
    main()
