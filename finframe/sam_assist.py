from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


class SamUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SamCapability:
    available: bool
    model_name: str
    message: str


@dataclass(frozen=True)
class SamMaskResult:
    mask: np.ndarray
    box: tuple[float, float, float, float]
    mask_rle: str
    confidence: float | None = None


def encode_mask_rle(mask: np.ndarray) -> str:
    """Encode a binary mask as an uncompressed COCO-compatible RLE string."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("A segmentation mask must be two-dimensional")
    flat = binary.flatten(order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        integer = int(bool(value))
        if integer == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = integer
    counts.append(run)
    return json.dumps(
        {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts},
        separators=(",", ":"),
    )


def decode_mask_rle(encoded: str) -> np.ndarray | None:
    if not encoded:
        return None
    try:
        payload = json.loads(encoded)
        height, width = (int(value) for value in payload["size"])
        counts = [int(value) for value in payload["counts"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if height <= 0 or width <= 0 or any(count < 0 for count in counts):
        return None
    expected = height * width
    if sum(counts) != expected:
        return None
    flat = np.zeros(expected, dtype=np.uint8)
    offset = 0
    foreground = False
    for count in counts:
        if foreground:
            flat[offset : offset + count] = 1
        offset += count
        foreground = not foreground
    return flat.reshape((height, width), order="F").astype(bool)


def mask_area_from_rle(encoded: str) -> int:
    mask = decode_mask_rle(encoded)
    return int(mask.sum()) if mask is not None else 0


def box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    binary = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(binary)
    if not len(columns):
        raise ValueError("SAM did not return a usable fish mask")
    height, width = binary.shape
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return (
        left / width,
        top / height,
        (right - left) / width,
        (bottom - top) / height,
    )


class SamAssistEngine:
    """Lazy, optional point-prompted segmentation through Ultralytics SAM."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser().resolve()
        configured = os.getenv("FINFRAME_SAM_MODEL", "").strip()
        model_dir = self.data_dir / "models"
        if configured:
            self.model_path = configured
        elif (model_dir / "sam3.pt").is_file():
            self.model_path = str(model_dir / "sam3.pt")
        elif (model_dir / "mobile_sam.pt").is_file():
            self.model_path = str(model_dir / "mobile_sam.pt")
        else:
            # Ultralytics downloads this lightweight checkpoint on first use.
            self.model_path = "mobile_sam.pt"
        filename = Path(self.model_path).name.casefold()
        self.model_name = "SAM 3" if filename.startswith("sam3") else "MobileSAM"
        self._model: Any | None = None
        self._lock = threading.Lock()

    def capability(self) -> SamCapability:
        try:
            import ultralytics
            from ultralytics import SAM  # noqa: F401
        except Exception as exc:
            return SamCapability(False, self.model_name, f"SAM unavailable: {exc}")
        if self.model_name == "SAM 3":
            version = tuple(
                int(part) for part in ultralytics.__version__.split(".")[:3] if part.isdigit()
            )
            if version < (8, 3, 237):
                return SamCapability(
                    False,
                    self.model_name,
                    "SAM 3 requires ultralytics 8.3.237 or newer",
                )
        configured = os.getenv("FINFRAME_SAM_MODEL", "").strip()
        if configured and not Path(configured).expanduser().is_file():
            return SamCapability(
                False,
                self.model_name,
                f"Configured SAM checkpoint was not found: {configured}",
            )
        detail = (
            f"{self.model_name} ready"
            if Path(self.model_path).expanduser().is_file()
            else "MobileSAM will download on first use"
        )
        return SamCapability(True, self.model_name, detail)

    def _load_model(self) -> Any:
        if self._model is None:
            capability = self.capability()
            if not capability.available:
                raise SamUnavailable(capability.message)
            try:
                from ultralytics import SAM

                self._model = SAM(self.model_path)
            except Exception as exc:
                raise SamUnavailable(f"Could not load {self.model_name}: {exc}") from exc
        return self._model

    def segment(
        self,
        image: np.ndarray,
        points: Sequence[tuple[float, float]],
        labels: Sequence[int],
    ) -> SamMaskResult:
        if not points or not any(int(label) == 1 for label in labels):
            raise ValueError("Add at least one positive point on the fish")
        height, width = image.shape[:2]
        pixel_points = [
            [max(0.0, min(width - 1.0, float(x) * width)), max(0.0, min(height - 1.0, float(y) * height))]
            for x, y in points
        ]
        device = os.getenv("FINFRAME_SAM_DEVICE", "").strip()
        options: dict[str, Any] = {"verbose": False}
        if device:
            options["device"] = device
        with self._lock:
            model = self._load_model()
            try:
                results = model.predict(
                    source=image,
                    points=pixel_points,
                    labels=[int(label) for label in labels],
                    **options,
                )
            except Exception as exc:
                raise SamUnavailable(f"{self.model_name} inference failed: {exc}") from exc
        if not results or results[0].masks is None or not len(results[0].masks.data):
            raise ValueError("SAM did not find an object at those points")
        masks = results[0].masks.data.detach().cpu().numpy()
        mask = np.asarray(masks[0] > 0.5, dtype=np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        confidence = None
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf):
            confidence = float(boxes.conf[0].detach().cpu())
        box = box_from_mask(mask)
        return SamMaskResult(
            mask=mask.astype(bool),
            box=box,
            mask_rle=encode_mask_rle(mask),
            confidence=confidence,
        )
