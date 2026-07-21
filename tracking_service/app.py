from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


TRACKER_FILES = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

app = FastAPI(
    title="FinFrame local tracking service",
    version="1.0.0",
    description="Runs a custom Ultralytics fish detector with ByteTrack or BoT-SORT on the local machine.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model: Any | None = None
_model_path: str | None = None


class Detection(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: dict[str, float]


class TrackedFrame(BaseModel):
    frame_number: int
    time_seconds: float
    detections: list[Detection]


class TrackingResponse(BaseModel):
    run_id: str
    tracker: str
    model: str
    video: dict[str, Any]
    classes: dict[int, str]
    frames: list[TrackedFrame]
    processed_frames: int
    returned_frames: int


def configured_model_path() -> Path | None:
    value = os.getenv("FINFRAME_MODEL_PATH", "").strip()
    return Path(value).expanduser().resolve() if value else None


def get_model() -> Any:
    global _model, _model_path
    model_path = configured_model_path()
    if model_path is None or not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Set FINFRAME_MODEL_PATH to trained fish-detector weights before running the tracking service.",
        )
    if _model is None or _model_path != str(model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="Install tracking_service/requirements.txt first.") from exc
        _model = YOLO(str(model_path))
        _model_path = str(model_path)
    return _model


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = configured_model_path()
    return {
        "ok": True,
        "model_ready": bool(model_path and model_path.is_file()),
        "model_path": model_path.name if model_path and model_path.is_file() else None,
        "trackers": sorted(TRACKER_FILES),
        "message": "Ready" if model_path and model_path.is_file() else "Service running; detector weights are not configured.",
    }


@app.post("/track", response_model=TrackingResponse)
def track_video(
    video: UploadFile = File(...),
    tracker: Literal["bytetrack", "botsort"] = Form("bytetrack"),
    confidence: float = Form(0.25, ge=0.01, le=0.99),
    sample_every: int = Form(5, ge=1, le=120),
    max_frames: int = Form(0, ge=0, le=1_000_000),
) -> TrackingResponse:
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix or 'unknown'}")
    model = get_model()
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(prefix="finframe_track_", suffix=suffix, delete=False) as target:
            shutil.copyfileobj(video.file, target)
            temporary_path = Path(target.name)

        results = model.track(
            source=str(temporary_path),
            stream=True,
            persist=True,
            tracker=TRACKER_FILES[tracker],
            conf=confidence,
            verbose=False,
        )
        frames: list[TrackedFrame] = []
        fps = 0.0
        width = 0
        height = 0
        processed = 0
        names = {int(key): str(value) for key, value in dict(model.names).items()}

        for frame_number, result in enumerate(results):
            processed = frame_number + 1
            if max_frames and processed > max_frames:
                break
            if not width or not height:
                height, width = map(int, result.orig_shape)
                fps = float(getattr(result, "speed", {}).get("fps", 0) or 0)
            if frame_number % sample_every:
                continue
            detections: list[Detection] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                xyxyn = boxes.xyxyn.cpu().tolist()
                for track_id, class_id, score, coords in zip(track_ids, class_ids, confidences, xyxyn, strict=True):
                    x1, y1, x2, y2 = map(float, coords)
                    detections.append(
                        Detection(
                            track_id=int(track_id),
                            class_id=int(class_id),
                            class_name=names.get(int(class_id), f"class_{class_id}"),
                            confidence=float(score),
                            bbox={"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)},
                        )
                    )
            frames.append(
                TrackedFrame(
                    frame_number=frame_number,
                    time_seconds=frame_number / fps if fps else 0.0,
                    detections=detections,
                )
            )

        # Ultralytics does not expose source FPS consistently on every result. Fall back to OpenCV metadata.
        if not fps:
            try:
                import cv2

                capture = cv2.VideoCapture(str(temporary_path))
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                if not width:
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                capture.release()
                if fps:
                    for frame in frames:
                        frame.time_seconds = frame.frame_number / fps
            except ImportError:
                pass

        return TrackingResponse(
            run_id=f"{tracker}-{uuid4().hex[:12]}",
            tracker=tracker,
            model=Path(_model_path or "model").name,
            video={"name": video.filename, "fps": fps, "width": width, "height": height},
            classes=names,
            frames=frames,
            processed_frames=processed,
            returned_frames=len(frames),
        )
    finally:
        video.file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
