from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


class AnnotationCanvas(QWidget):
    boxCreated = Signal(tuple)
    boxChanged = Signal(str, tuple)
    selectionChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(720, 420)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._annotations: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._action: str | None = None
        self._start = QPointF()
        self._current = QPointF()
        self._origin: tuple[float, float, float, float] | None = None

    def set_frame(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_annotations(self, annotations: list[dict[str, Any]]) -> None:
        self._annotations = annotations
        if self._selected_id and not any(item["id"] == self._selected_id for item in annotations):
            self._selected_id = None
        self.update()

    def select_annotation(self, annotation_id: str | None) -> None:
        self._selected_id = annotation_id
        self.update()

    def _image_rect(self) -> QRectF:
        if not self._image or self._image.isNull():
            return QRectF(self.rect())
        width, height = self.width(), self.height()
        scale = min(width / self._image.width(), height / self._image.height())
        draw_width, draw_height = self._image.width() * scale, self._image.height() * scale
        return QRectF((width - draw_width) / 2, (height - draw_height) / 2, draw_width, draw_height)

    def _normalised_point(self, position: QPointF) -> QPointF | None:
        target = self._image_rect()
        if not target.contains(position):
            return None
        return QPointF((position.x() - target.x()) / target.width(), (position.y() - target.y()) / target.height())

    @staticmethod
    def _box(annotation: dict[str, Any]) -> tuple[float, float, float, float]:
        return float(annotation["x"]), float(annotation["y"]), float(annotation["width"]), float(annotation["height"])

    def _screen_box(self, box: tuple[float, float, float, float]) -> QRectF:
        target = self._image_rect()
        x, y, width, height = box
        return QRectF(target.x() + x * target.width(), target.y() + y * target.height(), width * target.width(), height * target.height())

    def _hit(self, point: QPointF) -> dict[str, Any] | None:
        for annotation in reversed(self._annotations):
            x, y, width, height = self._box(annotation)
            if x <= point.x() <= x + width and y <= point.y() <= y + height:
                return annotation
        return None

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#071310"))
        target = self._image_rect()
        if self._image and not self._image.isNull():
            painter.drawImage(target, self._image)
        else:
            painter.setPen(QColor("#8ba49c"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a project and video to begin")
            return

        for annotation in self._annotations:
            selected = annotation["id"] == self._selected_id
            color = QColor(annotation.get("color") or "#ff8465")
            pen = QPen(color, 3 if selected else 2)
            if annotation.get("status") == "pending":
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            screen_box = self._screen_box(self._box(annotation))
            painter.fillRect(screen_box, QColor(color.red(), color.green(), color.blue(), 35))
            painter.drawRect(screen_box)
            label = f"{'AI? · ' if annotation.get('status') == 'pending' else ''}{annotation.get('code','')} · {annotation.get('track_id','')}"
            metrics = painter.fontMetrics()
            label_rect = QRectF(screen_box.x(), max(target.y(), screen_box.y() - 22), metrics.horizontalAdvance(label) + 12, 21)
            painter.fillRect(label_rect, color)
            painter.setPen(QColor("#06130f"))
            painter.drawText(label_rect.adjusted(6, 0, -3, 0), Qt.AlignmentFlag.AlignVCenter, label)
            if selected:
                painter.setBrush(QColor("white"))
                painter.setPen(QPen(color, 2))
                painter.drawRect(QRectF(screen_box.right() - 5, screen_box.bottom() - 5, 10, 10))

        if self._action == "draw":
            left, right = sorted((self._start.x(), self._current.x()))
            top, bottom = sorted((self._start.y(), self._current.y()))
            painter.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self._screen_box((left, top, right - left, bottom - top)))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._image:
            return
        point = self._normalised_point(event.position())
        if point is None:
            return
        hit = self._hit(point)
        self._start = point
        self._current = point
        if hit:
            self._selected_id = hit["id"]
            self._origin = self._box(hit)
            x, y, width, height = self._origin
            near_handle = abs(point.x() - (x + width)) < 0.025 and abs(point.y() - (y + height)) < 0.025
            self._action = "resize" if near_handle else "move"
            self.selectionChanged.emit(self._selected_id)
        else:
            self._selected_id = None
            self._origin = None
            self._action = "draw"
            self.selectionChanged.emit(None)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._action:
            return
        point = self._normalised_point(event.position())
        if point is None:
            return
        self._current = point
        if self._action in {"move", "resize"} and self._origin and self._selected_id:
            annotation = next((item for item in self._annotations if item["id"] == self._selected_id), None)
            if annotation:
                x, y, width, height = self._origin
                dx, dy = point.x() - self._start.x(), point.y() - self._start.y()
                if self._action == "move":
                    annotation["x"] = max(0.0, min(1.0 - width, x + dx))
                    annotation["y"] = max(0.0, min(1.0 - height, y + dy))
                else:
                    annotation["width"] = max(0.005, min(1.0 - x, width + dx))
                    annotation["height"] = max(0.005, min(1.0 - y, height + dy))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._action:
            return
        action = self._action
        self._action = None
        if action == "draw":
            left, right = sorted((self._start.x(), self._current.x()))
            top, bottom = sorted((self._start.y(), self._current.y()))
            if right - left >= 0.008 and bottom - top >= 0.008:
                self.boxCreated.emit((left, top, right - left, bottom - top))
        elif self._selected_id:
            annotation = next((item for item in self._annotations if item["id"] == self._selected_id), None)
            if annotation:
                self.boxChanged.emit(self._selected_id, self._box(annotation))
        self.update()
