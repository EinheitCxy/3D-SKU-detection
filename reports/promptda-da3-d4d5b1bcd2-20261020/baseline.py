"""Run the existing low-overhead Scanner profiler on this 20-frame recording.

The upstream report tool was specific to a 14-frame fixture. Adapt only its two
frame-count guards in memory; candidate loading/prediction/export stay native.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source_path = ROOT / "scripts/3d/scanner/profile_candidates.py"
source = source_path.read_text()
edits = {
    'if len(images) != 14 or len(sensor["frame_ids"]) != 14:':
        'if not images or len(sensor["frame_ids"]) != len(images):',
    'depth.shape != (14, expected_height, expected_width)':
        'depth.shape != (len(images), expected_height, expected_width)',
    'This comparison requires the same prepared 14-frame scan':
        'Prepared RGB and sensor frame counts must agree',
}
for old, new in edits.items():
    if source.count(old) != 1:
        raise RuntimeError(f"Profiler source changed at expected fixture guard: {old}")
    source = source.replace(old, new)
import sys
sys.path.insert(0, str(source_path.parent))
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
