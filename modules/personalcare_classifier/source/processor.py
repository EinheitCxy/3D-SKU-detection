from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.mobilenet import mobilenet_v3_large

_INPUT_SIZE = (160, 160)
_IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
_BIT_REVERSE = bytes(int(f"{value:08b}"[::-1], 2) for value in range(256))


class PersonalcarePredictor:
    def __init__(self, device: str) -> None:
        self.device = self._require_cuda_device(device)
        source_dir = Path(__file__).resolve().parent
        model_dir = source_dir / "model"
        self.project_id = self._read_project_id(model_dir / "info.json")
        self.class_names = self._read_class_names(model_dir / "classnames.txt")
        self.model = mobilenet_v3_large(weights=None, num_classes=len(self.class_names))
        self.model.load_state_dict(self._load_state_dict(model_dir / "model.bin"))
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _require_cuda_device(device: str) -> torch.device:
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("personalcare classifier requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device is unavailable: {device}")
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device is unavailable: {device}")
        return requested

    @staticmethod
    def _read_project_id(info_path: Path) -> int:
        project_id = json.loads(info_path.read_text(encoding="utf-8"))["project_id"]
        if isinstance(project_id, bool) or not isinstance(project_id, int):
            raise ValueError("model project_id must be an integer")
        return project_id

    @staticmethod
    def _read_class_names(class_names_path: Path) -> list[str]:
        class_names = class_names_path.read_text(encoding="utf-8").splitlines()
        if not class_names:
            raise ValueError("model class names must not be empty")
        return class_names

    @staticmethod
    def _load_state_dict(model_path: Path) -> dict[str, object]:
        encoded = bytearray(model_path.read_bytes())
        decoded = bytearray(encoded.translate(_BIT_REVERSE))
        decoded.reverse()
        checkpoint = torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("personalcare model checkpoint has no state_dict")
        return state_dict

    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        predictions: list[tuple[str, float]] = []
        with torch.inference_mode():
            for start in range(0, len(crops), 32):
                batch = self._prepare_batch(crops[start : start + 32]).to(self.device)
                confidences, indices = self.model(batch).softmax(dim=1).topk(1, dim=1)
                predictions.extend(
                    (self.class_names[index], float(confidence))
                    for index, confidence in zip(
                        indices.flatten().tolist(), confidences.flatten().tolist()
                    )
                )
        return predictions

    @staticmethod
    def _prepare_batch(crops: list[np.ndarray]) -> torch.Tensor:
        if not crops:
            return torch.empty((0, 3, *_INPUT_SIZE), dtype=torch.float32)
        return torch.from_numpy(np.stack([_prepare_crop(crop) for crop in crops]))


def _prepare_crop(crop: np.ndarray) -> np.ndarray:
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("crops must be OpenCV BGR uint8 images")
    height, width = crop.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("crops must not be empty")
    ratio = max(_INPUT_SIZE) / max(width, height)
    resized_width = min(_INPUT_SIZE[0], max(1, int(width * ratio)))
    resized_height = min(_INPUT_SIZE[1], max(1, int(height * ratio)))
    resized = cv2.resize(crop, (resized_width, resized_height))
    normalized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (normalized - _IMAGENET_MEAN) / _IMAGENET_STD
    pad_width = _INPUT_SIZE[0] - resized_width
    pad_height = _INPUT_SIZE[1] - resized_height
    padded = cv2.copyMakeBorder(
        normalized,
        pad_height // 2,
        pad_height - pad_height // 2,
        pad_width // 2,
        pad_width - pad_width // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
