from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

from .database import Database
from .tracking_identity import BoundaryIdentityAllocator


class InferenceError(RuntimeError):
    pass


class InferenceEngine:
    """Loads the active validated detector and emits unverified proposals."""

    def __init__(self, db: Database):
        self.db = db
        self._model: Any | None = None
        self._model_id: str | None = None
        self._lock = threading.Lock()

    def active_model(self) -> dict[str, Any]:
        active = self.db.active_model()
        if not active or not Path(active["weights_path"]).is_file():
            raise InferenceError("No validated detector is active yet")
        return active

    def _load(self) -> tuple[Any, dict[str, Any]]:
        active = self.active_model()
        with self._lock:
            if self._model is None or self._model_id != active["id"]:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:
                    raise InferenceError("Install Ultralytics to enable AI suggestions") from exc
                self._model = YOLO(active["weights_path"])
                self._model_id = active["id"]
        return self._model, active

    def _species_for_name(self, name: str) -> dict[str, Any] | None:
        by_code = self.db.species_by_code(name)
        if by_code:
            return by_code
        normalised = name.strip().lower()
        return next((item for item in self.db.list_species() if normalised in {item["common_name"].lower(), item["scientific_name"].lower()}), None)

    def detect_frame(self, image: Any, *, confidence: float = 0.25) -> list[dict[str, Any]]:
        model, active = self._load()
        results = model.predict(source=image, conf=confidence, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        names = {int(key): str(value) for key, value in dict(model.names).items()}
        proposals = []
        for class_id, score, coords in zip(boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxyn.cpu().tolist(), strict=True):
            species = self._species_for_name(names.get(int(class_id), str(class_id)))
            if species is None:
                continue
            x1, y1, x2, y2 = map(float, coords)
            proposals.append({
                "species_id": species["id"],
                "box": (x1, y1, max(0.0001, x2 - x1), max(0.0001, y2 - y1)),
                "confidence": float(score),
                "model_id": active["id"],
                "source": "ai",
                "track_id": None,
            })
        return proposals

    def classify_box(self, image: Any, box: tuple[float, float, float, float], *, confidence: float = 0.05) -> dict[str, Any] | None:
        """Suggest a species for a student-drawn crop without accepting the label."""
        model, active = self._load()
        image_height, image_width = image.shape[:2]
        x, y, width, height = box
        left = max(0, min(image_width - 1, round(x * image_width)))
        top = max(0, min(image_height - 1, round(y * image_height)))
        right = max(left + 1, min(image_width, round((x + width) * image_width)))
        bottom = max(top + 1, min(image_height, round((y + height) * image_height)))
        crop = image[top:bottom, left:right]
        results = model.predict(source=crop, conf=confidence, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        scores = boxes.conf.cpu().tolist()
        best_index = max(range(len(scores)), key=scores.__getitem__)
        class_id = int(boxes.cls.int().cpu().tolist()[best_index])
        species = self._species_for_name(str(dict(model.names).get(class_id, class_id)))
        if species is None:
            return None
        return {
            "species_id": species["id"],
            "confidence": float(scores[best_index]),
            "model_id": active["id"],
        }

    def track_video(
        self,
        video_path: str,
        *,
        tracker: str = "bytetrack",
        confidence: float = 0.25,
        sample_every: int = 1,
    ) -> Iterator[dict[str, Any]]:
        if tracker not in {"bytetrack", "botsort"}:
            raise ValueError("tracker must be bytetrack or botsort")
        model, active = self._load()
        names = {int(key): str(value) for key, value in dict(model.names).items()}
        identities = BoundaryIdentityAllocator("BT" if tracker == "bytetrack" else "BS")
        results = model.track(
            source=video_path,
            tracker=f"{tracker}.yaml",
            conf=confidence,
            stream=True,
            persist=True,
            verbose=False,
        )
        for frame_number, result in enumerate(results):
            if frame_number % max(1, sample_every):
                continue
            boxes = result.boxes
            raw_detections: list[dict[str, Any]] = []
            if boxes is not None and boxes.id is not None:
                for track_id, class_id, score, coords in zip(
                    boxes.id.int().cpu().tolist(),
                    boxes.cls.int().cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    boxes.xyxyn.cpu().tolist(),
                    strict=True,
                ):
                    species = self._species_for_name(names.get(int(class_id), str(class_id)))
                    if species is None:
                        continue
                    x1, y1, x2, y2 = map(float, coords)
                    raw_detections.append({
                        "species_id": species["id"],
                        "box": (x1, y1, max(0.0001, x2 - x1), max(0.0001, y2 - y1)),
                        "confidence": float(score),
                        "model_id": active["id"],
                        "source": "tracker",
                        "raw_track_id": int(track_id),
                    })
            detections = identities.assign(frame_number, raw_detections)
            yield {"frame_number": frame_number, "detections": detections}
