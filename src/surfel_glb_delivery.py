"""Compact GLB delivery from a fixed Viewer generation or published CURRENT."""
import argparse
import json
from pathlib import Path

from src.surfel_glb import export_scene_glb


def current_generation(root):
    root = Path(root).resolve()
    pointer = json.loads((root / 'CURRENT').read_text(encoding='utf-8'))
    run_id = pointer.get('run_id') if isinstance(pointer, dict) else None
    if not isinstance(run_id, str) or not run_id or run_id in ('.', '..') or Path(run_id).name != run_id:
        raise ValueError('Invalid Viewer CURRENT run_id')
    generation = (root / 'runs' / run_id).resolve()
    if not generation.is_relative_to(root / 'runs') or not generation.is_dir():
        raise ValueError('Viewer CURRENT generation missing or outside runs')
    return generation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--generation', type=Path)
    source.add_argument('--viewer-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bake-device', default='cpu', help='cpu or logical cuda:N; no automatic fallback')
    parser.add_argument('--encoder-workers', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--bake-fixed-shape', action='store_true', help='Experimental fixed-shape CUDA sampler')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    generation = current_generation(args.viewer_root) if args.viewer_root is not None else args.generation
    report = export_scene_glb(generation, args.output, bake_device=args.bake_device,
                              bake_fixed_shape=args.bake_fixed_shape,
                              encoder_workers=args.encoder_workers)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report is not None:
        with args.report.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
