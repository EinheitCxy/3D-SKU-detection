"""Pack a Viewer generation using the mapping service's flat ZIP contract."""

import json
from pathlib import Path
import zipfile

_FILES = (
    "manifest.json", "positions.f32.bin", "colors.u8.bin", "normals.i8.bin",
    "objects.json", "surfel.json", "surfel-u.f16.bin", "surfel-v.f16.bin",
    "surfel-frame.u8.bin", "surfel-depth.f16.bin",
)


def pack_bundle(generation_dir: Path, out_zip: Path) -> int:
    generation_dir, out_zip = Path(generation_dir), Path(out_zip)
    surfel = json.loads((generation_dir / "surfel.json").read_text(encoding="utf-8"))
    if surfel.get("version") not in (2, 3):
        raise ValueError("Unsupported Surfel version")
    members = [*_FILES, *(frame["texture"] for frame in surfel["frames"])]
    if surfel["version"] == 3:
        members.append("surfel-scale.f16.bin")
    members.extend(f"thumbs/{p.name}" for p in sorted((generation_dir / "thumbs").glob("*.jpg")))
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in members:
            archive.write(
                generation_dir / name, name,
                compress_type=zipfile.ZIP_DEFLATED if name.endswith(".bin") else zipfile.ZIP_STORED,
                compresslevel=1,
            )
    return out_zip.stat().st_size
