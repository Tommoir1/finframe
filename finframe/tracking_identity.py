from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NormalisedBox = tuple[float, float, float, float]


def touches_boundary(box: NormalisedBox, margin: float = 0.015) -> bool:
    x, y, width, height = box
    return (
        x <= margin
        or y <= margin
        or x + width >= 1 - margin
        or y + height >= 1 - margin
    )


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

    def assign(
        self,
        frame_number: int,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
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
