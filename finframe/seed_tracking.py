from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


NormalisedBox = tuple[float, float, float, float]
PixelBox = tuple[int, int, int, int]
TrackerFactory = Callable[[], Any]


class SeedTrackingUnavailable(RuntimeError):
    pass


def create_opencv_tracker() -> Any:
    """Create the strongest locally available CPU box tracker."""
    try:
        import cv2
    except ImportError as exc:
        raise SeedTrackingUnavailable("OpenCV is required for box-seeded tracking") from exc

    candidates = (
        (cv2, "TrackerCSRT_create"),
        (cv2, "TrackerKCF_create"),
        (getattr(cv2, "legacy", None), "TrackerCSRT_create"),
        (getattr(cv2, "legacy", None), "TrackerKCF_create"),
        (cv2, "TrackerMIL_create"),
    )
    for namespace, name in candidates:
        factory = getattr(namespace, name, None) if namespace is not None else None
        if factory is not None:
            return factory()
    raise SeedTrackingUnavailable(
        "This OpenCV installation has no supported box tracker. Install opencv-contrib-python."
    )


def touches_boundary(box: NormalisedBox, margin: float = 0.015) -> bool:
    x, y, width, height = box
    return x <= margin or y <= margin or x + width >= 1 - margin or y + height >= 1 - margin


def _normalised_to_pixels(box: NormalisedBox, image: Any) -> PixelBox:
    height, width = image.shape[:2]
    x, y, box_width, box_height = box
    return (
        round(x * width),
        round(y * height),
        max(2, round(box_width * width)),
        max(2, round(box_height * height)),
    )


def _pixels_to_visible_box(box: PixelBox, image: Any) -> tuple[NormalisedBox | None, float]:
    image_height, image_width = image.shape[:2]
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        return None, 0.0
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(image_width), x + width), min(float(image_height), y + height)
    visible_width, visible_height = max(0.0, right - left), max(0.0, bottom - top)
    visible_ratio = (visible_width * visible_height) / (width * height)
    if visible_width < 2 or visible_height < 2:
        return None, visible_ratio
    return (
        left / image_width,
        top / image_height,
        visible_width / image_width,
        visible_height / image_height,
    ), visible_ratio


@dataclass
class SeedTrack:
    track_id: str
    species_id: str
    source_annotation_id: str
    tracker: Any
    last_frame: int
    last_box: NormalisedBox
    life_stage: str
    activity: str
    uncertain: bool
    misses: int = 0


@dataclass(frozen=True)
class SeedPrediction:
    track_id: str
    species_id: str
    source_annotation_id: str
    frame_number: int
    box: NormalisedBox
    life_stage: str
    activity: str
    uncertain: bool


@dataclass(frozen=True)
class EndedTrack:
    track_id: str
    reason: str


class SeedTrackingSession:
    """Propagate student boxes while preserving a conservative identity boundary."""

    def __init__(
        self,
        tracker_factory: TrackerFactory = create_opencv_tracker,
        *,
        boundary_margin: float = 0.015,
        minimum_visible_ratio: float = 0.35,
        maximum_interior_misses: int = 5,
    ):
        self.tracker_factory = tracker_factory
        self.boundary_margin = boundary_margin
        self.minimum_visible_ratio = minimum_visible_ratio
        self.maximum_interior_misses = maximum_interior_misses
        self._tracks: dict[str, SeedTrack] = {}

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def clear(self) -> None:
        self._tracks.clear()

    def stop(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)

    def seed(self, annotation: dict[str, Any], image: Any, frame_number: int) -> None:
        tracker = self.tracker_factory()
        normalised_box = (
            float(annotation["x"]),
            float(annotation["y"]),
            float(annotation["width"]),
            float(annotation["height"]),
        )
        initialised = tracker.init(image, _normalised_to_pixels(normalised_box, image))
        if initialised is False:
            raise SeedTrackingUnavailable("The box tracker could not initialise from this annotation")
        track_id = str(annotation["track_id"])
        self._tracks[track_id] = SeedTrack(
            track_id=track_id,
            species_id=str(annotation["species_id"]),
            source_annotation_id=str(annotation["id"]),
            tracker=tracker,
            last_frame=int(frame_number),
            last_box=normalised_box,
            life_stage=str(annotation.get("life_stage", "Unknown")),
            activity=str(annotation.get("activity", "Unknown")),
            uncertain=bool(annotation.get("uncertain", False)),
        )

    def update(self, image: Any, frame_number: int) -> tuple[list[SeedPrediction], list[EndedTrack]]:
        predictions: list[SeedPrediction] = []
        ended: list[EndedTrack] = []
        for track_id, state in list(self._tracks.items()):
            if frame_number != state.last_frame + 1:
                ended.append(EndedTrack(track_id, "non_sequential_seek"))
                self._tracks.pop(track_id, None)
                continue
            ok, pixel_box = state.tracker.update(image)
            if not ok:
                state.last_frame = int(frame_number)
                state.misses += 1
                if touches_boundary(state.last_box, self.boundary_margin):
                    ended.append(EndedTrack(track_id, "left_frame"))
                    self._tracks.pop(track_id, None)
                elif state.misses >= self.maximum_interior_misses:
                    ended.append(EndedTrack(track_id, "lost_in_frame"))
                    self._tracks.pop(track_id, None)
                continue

            box, visible_ratio = _pixels_to_visible_box(pixel_box, image)
            if box is None or visible_ratio < self.minimum_visible_ratio:
                ended.append(EndedTrack(track_id, "left_frame"))
                self._tracks.pop(track_id, None)
                continue

            state.last_frame = int(frame_number)
            state.last_box = box
            state.misses = 0
            predictions.append(SeedPrediction(
                track_id=track_id,
                species_id=state.species_id,
                source_annotation_id=state.source_annotation_id,
                frame_number=int(frame_number),
                box=box,
                life_stage=state.life_stage,
                activity=state.activity,
                uncertain=state.uncertain,
            ))
        return predictions, ended


@dataclass
class _IdentityState:
    public_id: str
    last_frame: int
    last_box: NormalisedBox
    retired: bool = False


class BoundaryIdentityAllocator:
    """Prevent detector trackers from reusing an identity after a boundary exit."""

    def __init__(self, prefix: str, *, boundary_margin: float = 0.015):
        self.prefix = prefix.upper()
        self.boundary_margin = boundary_margin
        self._next_identity = 1
        self._states: dict[int, _IdentityState] = {}

    def _new_identity(self) -> str:
        public_id = f"{self.prefix}-{self._next_identity:05d}"
        self._next_identity += 1
        return public_id

    def assign(self, frame_number: int, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current_raw_ids = {int(item["raw_track_id"]) for item in detections}
        for raw_id, state in self._states.items():
            if raw_id not in current_raw_ids and state.last_frame < frame_number:
                if touches_boundary(state.last_box, self.boundary_margin):
                    state.retired = True

        assigned: list[dict[str, Any]] = []
        for detection in detections:
            raw_id = int(detection["raw_track_id"])
            box = tuple(map(float, detection["box"]))
            state = self._states.get(raw_id)
            if state is None or state.retired:
                state = _IdentityState(self._new_identity(), int(frame_number), box)
                self._states[raw_id] = state
            else:
                state.last_frame = int(frame_number)
                state.last_box = box
            result = dict(detection)
            result.pop("raw_track_id", None)
            result["track_id"] = state.public_id
            assigned.append(result)
        return assigned
