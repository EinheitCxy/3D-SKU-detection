"""Export an independent Pi3X comparison bundle from existing matching/cache."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deduplicate_detections import deduplicate_sequence, resolve_dataset_paths
from src.web_viewer_export import export_web_viewer_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('Output'))
    parser.add_argument('--viewer-output', type=Path, default=Path('modules/viewer_web/public/data-pi3x'))
    parser.add_argument('--sku-masterdata-csv', type=Path, default=Path('runtime/sku_masterdata.csv'))
    args = parser.parse_args()
    output = args.output_root / args.dataset.name
    classification = output / 'personalcare_classification'
    current = json.loads((classification / 'CURRENT').read_text())
    if current['complete'] is not True:
        raise ValueError('SKU classification is incomplete')
    detections = classification / 'runs' / current['run_id'] / 'detections'
    deduplicate_sequence(
        resolve_dataset_paths(args.dataset, detections), output_root=args.output_root,
        output_subdir='dedup_detections_pi3x', algorithm='3d_mapping',
        backend='pi3x', min_hit_ratio=0.0, same_names=True,
    )
    result = export_web_viewer_bundle(
        dataset_name=args.dataset.name,
        da3_cache_path=output / 'pi3x_cache/predictions.npz', backend='Pi3X',
        global_mapping_path=output / 'dedup_detections_pi3x/global_mapping.json',
        output_dir=args.viewer_output, source_images_dir=args.dataset / 'images',
        sam3_mask_cache_root=output / 'sam3_mask_cache/v2',
        sku_masterdata_csv=args.sku_masterdata_csv,
        voxel_size_m=0.005, max_points=1500000,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
