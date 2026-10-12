"""Build registered arm comparison sheets from capture and score artifacts."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheet(inputs: Path, output_dir: Path) -> list[Path]:
    outputs = []
    registered_scenes = {"video3", "fd6"}
    registered_arms = {"V0", "V1a", "V1b", "V2", "X0", "XQ", "XS4", "XS8", "XQS", "G0", "G1", "G2", "G3"}
    for scene_dir in sorted(p for p in inputs.iterdir() if p.is_dir() and p.name in registered_scenes):
        arms = []
        for arm_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name in registered_arms):
            capture = arm_dir / "capture"
            if not capture.is_dir():
                continue
            capture_receipt = capture / "capture.json"
            if not capture_receipt.exists():
                continue
            receipt = json.loads(capture_receipt.read_text())
            required_views = {"overview", "close", "side", "selected"}
            if receipt.get("failure") or not required_views.issubset(receipt.get("screenshots", {})):
                continue
            score_paths = list((arm_dir / "refview").glob("*/score.json")) if (arm_dir / "refview").is_dir() else []
            matching_scores = [path for path in score_paths if path.parent.name == arm_dir.name]
            score_path = matching_scores[0] if len(matching_scores) == 1 else score_paths[0] if len(score_paths) == 1 else None
            score = json.loads(score_path.read_text()) if score_path is not None else {}
            values = score.get("mean", {})
            label = f"{arm_dir.name}  PSNR {values.get('PSNR', float('nan')):.2f} dB  coverage {values.get('coverage', float('nan')):.3f}"
            arms.append((arm_dir.name, label, capture))
        for view in ("overview", "close", "side", "selected"):
            panels = [(name, label, capture / f"{view}.png") for name, label, capture in arms if (capture / f"{view}.png").exists()]
            if not panels: continue
            cell_width, cell_height, header, gap = 480, 270, 58, 10
            sheet = Image.new("RGB",(len(panels)*(cell_width+gap)-gap,header+cell_height),(245,245,245)); draw=ImageDraw.Draw(sheet)
            for column,(name,label,path) in enumerate(panels):
                x=column*(cell_width+gap)
                draw.text((x+8,5),name,fill=(0,0,0))
                draw.text((x+8,23),label.replace(f"{name}  ",""),fill=(0,0,0))
                with Image.open(path) as source:
                    image=source.convert("RGB").resize((cell_width,cell_height))
                sheet.paste(image,(x,header))
            output_dir.mkdir(parents=True,exist_ok=True)
            target=output_dir/f"{scene_dir.name}-{view}.png"; sheet.save(target); outputs.append(target)
    return outputs


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--inputs",required=True,type=Path); parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    for path in make_sheet(args.inputs,args.out_dir): print(path)


if __name__ == "__main__": main()
