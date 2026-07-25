from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .sam_assist import decode_mask_rle


class AnnotationCanvas(QWidget):
    boxCreated = Signal(tuple)
    boxChanged = Signal(str, tuple)
    selectionChanged = Signal(object)
    samPointCreated = Signal(tuple, int)

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
        self._annotation_masks: dict[str, QImage] = {}
        self._sam_enabled = False
        self._sam_preview: QImage | None = None
        self._sam_points: list[tuple[float, float, int]] = []

    def set_frame(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_annotations(self, annotations: list[dict[str, Any]]) -> None:
        self._annotations = annotations
        self._annotation_masks = {}
        for annotation in annotations:
            mask = decode_mask_rle(str(annotation.get("mask_rle") or ""))
            if mask is not None:
                color = QColor(annotation.get("color") or "#ff8465")
                self._annotation_masks[annotation["id"]] = self._mask_image(mask, color, 58)
        if self._selected_id and not any(item["id"] == self._selected_id for item in annotations):
            self._selected_id = None
        self.update()

    @staticmethod
    def _mask_image(mask: np.ndarray, color: QColor, alpha: int) -> QImage:
        binary = np.asarray(mask, dtype=bool)
        height, width = binary.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[binary, 0] = color.red()
        rgba[binary, 1] = color.green()
        rgba[binary, 2] = color.blue()
        rgba[binary, 3] = alpha
        return QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()

    def set_sam_mode(self, enabled: bool) -> None:
        self._sam_enabled = enabled
        self._action = None
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def set_sam_preview(
        self,
        mask: np.ndarray | None,
        points: list[tuple[float, float, int]],
    ) -> None:
        self._sam_preview = (
            self._mask_image(mask, QColor("#22d3ee"), 92) if mask is not None else None
        )
        self._sam_points = list(points)
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

        if self._sam_preview is not None:
            painter.drawImage(target, self._sam_preview)

        for annotation in self._annotations:
            selected = annotation["id"] == self._selected_id
            color = QColor(annotation.get("color") or "#ff8465")
            mask_image = self._annotation_masks.get(annotation["id"])
            if mask_image is not None:
                painter.drawImage(target, mask_image)
            pen = QPen(color, 3 if selected else 2)
            if annotation.get("status") == "pending":
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            screen_box = self._screen_box(self._box(annotation))
            painter.fillRect(screen_box, QColor(color.red(), color.green(), color.blue(), 35 if selected else 14))
            painter.drawRect(screen_box)
            label = f"{'AI? · ' if annotation.get('status') == 'pending' else ''}{annotation.get('code','')} · {annotation.get('track_id','')}"
            label_font = painter.font()
            label_font.setPixelSize(11 if selected else 9)
            label_font.setBold(selected)
            painter.setFont(label_font)
            metrics = painter.fontMetrics()
            label_height = 17 if selected else 13
            horizontal_padding = 5 if selected else 3
            label_rect = QRectF(
                screen_box.x(),
                max(target.y(), screen_box.y() - label_height - 1),
                metrics.horizontalAdvance(label) + horizontal_padding * 2,
                label_height,
            )
            label_background = QColor(color)
            label_background.setAlpha(235 if selected else 105)
            painter.fillRect(label_rect, label_background)
            label_text = QColor("#06130f")
            label_text.setAlpha(255 if selected else 205)
            painter.setPen(label_text)
            painter.drawText(
                label_rect.adjusted(horizontal_padding, 0, -horizontal_padding, 0),
                Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            if selected:
                painter.setBrush(QColor("white"))
                painter.setPen(QPen(color, 2))
                painter.drawRect(QRectF(screen_box.right() - 5, screen_box.bottom() - 5, 10, 10))

        for x, y, label in self._sam_points:
            centre = QPointF(target.x() + x * target.width(), target.y() + y * target.height())
            color = QColor("#25d366") if label else QColor("#ff4d5a")
            painter.setBrush(color)
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawEllipse(centre, 6, 6)
            painter.setPen(QPen(QColor("#071310"), 2))
            painter.drawLine(QLineF(centre.x() - 3, centre.y(), centre.x() + 3, centre.y()))
            if label:
                painter.drawLine(QLineF(centre.x(), centre.y() - 3, centre.x(), centre.y() + 3))

        if self._action == "draw":
            left, right = sorted((self._start.x(), self._current.x()))
            top, bottom = sorted((self._start.y(), self._current.y()))
            painter.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self._screen_box((left, top, right - left, bottom - top)))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._image:
            return
        point = self._normalised_point(event.position())
        if point is None:
            return
        if self._sam_enabled:
            if event.button() not in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
                return
            negative = (
                event.button() == Qt.MouseButton.RightButton
                or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            )
            self.samPointCreated.emit((point.x(), point.y()), 0 if negative else 1)
            return
        if event.button() != Qt.MouseButton.LeftButton:
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
