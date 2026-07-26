from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
from PySide6.QtCore import QSize, QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .canvas import AnnotationCanvas
from .database import Database
from .dataset import (
    DatasetError,
    export_contribution_bundle,
    export_dataset,
    export_project_backup,
    import_contribution_bundle,
)
from .inference import InferenceEngine, InferenceError
from .sam_assist import SamAssistEngine, SamMaskResult
from .training import TrainingCoordinator


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
IMAGE_FILE_FILTER = "Image files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)"


def discover_image_files(folder: str | Path) -> list[Path]:
    """Return supported images in a folder and its subfolders in stable order."""
    root = Path(folder).expanduser().resolve()
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).casefold(),
    )


def _iou(first: dict[str, Any], second: tuple[float, float, float, float]) -> float:
    x, y, width, height = second
    left, top = max(float(first["x"]), x), max(float(first["y"]), y)
    right = min(float(first["x"]) + float(first["width"]), x + width)
    bottom = min(float(first["y"]) + float(first["height"]), y + height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = float(first["width"]) * float(first["height"]) + width * height - intersection
    return intersection / union if union else 0.0


def suggested_species_code(name: str) -> str:
    """Create a readable, deterministic class code for a new taxon."""
    tokens = re.findall(r"[A-Za-z0-9]+", name.upper())
    return f"USR-{'-'.join(tokens)}" if tokens else "USR-SPECIES"


def navigation_arrow_icon(direction: int) -> QIcon:
    """Draw a high-resolution arrow instead of using a platform bitmap icon."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(
        QPen(
            QColor("#173b30"),
            6,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    points = (
        ((40, 13), (21, 32), (40, 51))
        if direction < 0
        else ((24, 13), (43, 32), (24, 51))
    )
    painter.drawLine(*points[0], *points[1])
    painter.drawLine(*points[1], *points[2])
    painter.end()
    return QIcon(pixmap)


class ProjectDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, project: dict[str, Any] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit survey project" if project else "New survey project")
        layout = QFormLayout(self)
        self.name = QLineEdit()
        self.deployment = QLineEdit()
        self.site = QLineEdit()
        self.observer = QLineEdit()
        self.survey_date = QLineEdit()
        self.depth = QLineEdit()
        self.notes = QTextEdit()
        layout.addRow("Project name", self.name)
        layout.addRow("Deployment ID", self.deployment)
        layout.addRow("Site", self.site)
        layout.addRow("Student / observer", self.observer)
        layout.addRow("Survey date", self.survey_date)
        layout.addRow("Depth", self.depth)
        layout.addRow("Notes", self.notes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        if project:
            self.name.setText(project.get("name", ""))
            self.deployment.setText(project.get("deployment_id", ""))
            self.site.setText(project.get("site", ""))
            self.observer.setText(project.get("observer", ""))
            self.survey_date.setText(project.get("survey_date", ""))
            self.depth.setText(project.get("depth", ""))
            self.notes.setPlainText(project.get("notes", ""))

    def values(self) -> dict[str, str]:
        return {
            "name": self.name.text().strip(),
            "deployment_id": self.deployment.text().strip(),
            "site": self.site.text().strip(),
            "observer": self.observer.text().strip(),
            "survey_date": self.survey_date.text().strip(),
            "depth": self.depth.text().strip(),
            "notes": self.notes.toPlainText().strip(),
        }


class SpeciesDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, species: dict[str, Any] | None = None):
        super().__init__(parent)
        self._editing = species is not None
        self._code_was_edited = False
        self._color = QColor(str(species.get("color", "#ff8465")) if species else "#ff8465")
        self.setWindowTitle("Edit species" if species else "Add new species")
        self.setMinimumWidth(440)

        layout = QFormLayout(self)
        self.common_name = QLineEdit()
        self.common_name.setPlaceholderText("e.g. Norfolk Chromis")
        self.scientific_name = QLineEdit()
        self.scientific_name.setPlaceholderText("e.g. Chromis norfolkensis")
        self.code = QLineEdit()
        self.code.setPlaceholderText("Generated from the scientific name")
        self.code.setToolTip(
            "This stable identifier is used in exported machine-learning datasets. "
            "It cannot be changed after the species is created."
        )
        self.color_button = QPushButton()
        self.color_button.clicked.connect(self.choose_color)
        self._refresh_color_button()

        layout.addRow("Common name", self.common_name)
        layout.addRow("Scientific name", self.scientific_name)
        layout.addRow("Dataset code", self.code)
        layout.addRow("Bounding-box colour", self.color_button)
        explanation = QLabel(
            "Names can be corrected later. The dataset code remains fixed so existing "
            "annotations and trained models continue to refer to the same class."
        )
        explanation.setWordWrap(True)
        layout.addRow(explanation)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        if species:
            self.common_name.setText(str(species.get("common_name", "")))
            self.scientific_name.setText(str(species.get("scientific_name", "")))
            self.code.setText(str(species.get("code", "")))
            self.code.setReadOnly(True)
        else:
            self.scientific_name.textEdited.connect(self.refresh_suggested_code)
            self.common_name.textEdited.connect(self.refresh_suggested_code)
            self.code.textEdited.connect(self.mark_code_edited)

    def mark_code_edited(self, _text: str) -> None:
        self._code_was_edited = True

    def refresh_suggested_code(self, _text: str = "") -> None:
        if self._editing or self._code_was_edited:
            return
        source = self.scientific_name.text().strip() or self.common_name.text().strip()
        self.code.setText(suggested_species_code(source))

    def choose_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Bounding-box colour")
        if color.isValid():
            self._color = color
            self._refresh_color_button()

    def _refresh_color_button(self) -> None:
        self.color_button.setText(f"Choose colour…  {self._color.name()}")
        self.color_button.setStyleSheet(f"background-color: {self._color.name()};")

    def validate_and_accept(self) -> None:
        if not self.common_name.text().strip():
            QMessageBox.warning(self, "Common name required", "Enter a common name.")
            return
        if not self.scientific_name.text().strip():
            QMessageBox.warning(self, "Scientific name required", "Enter a scientific name.")
            return
        if not self.code.text().strip():
            QMessageBox.warning(self, "Dataset code required", "Enter a stable dataset code.")
            return
        self.accept()

    def values(self) -> dict[str, str]:
        return {
            "common_name": self.common_name.text().strip(),
            "scientific_name": self.scientific_name.text().strip(),
            "code": self.code.text().strip().upper(),
            "color": self._color.name(),
        }


class TrackingWorker(QThread):
    progress = Signal(int, str)
    failed = Signal(str)
    completed = Signal(int)

    def __init__(self, db: Database, engine: InferenceEngine, video: dict[str, Any], tracker: str, confidence: float, sample_every: int):
        super().__init__()
        self.db, self.engine, self.video = db, engine, video
        self.tracker, self.confidence, self.sample_every = tracker, confidence, sample_every

    def run(self) -> None:
        added = 0
        try:
            self.db.delete_pending_proposals(self.video["id"], source="tracker", model_only=True)
            total = max(1, int(self.video["frame_count"]))
            for tracked in self.engine.track_video(self.video["path"], tracker=self.tracker, confidence=self.confidence, sample_every=self.sample_every):
                frame_number = int(tracked["frame_number"])
                existing = self.db.annotations_for_frame(self.video["id"], frame_number)
                for proposal in tracked["detections"]:
                    duplicate = any(
                        item["status"] == "verified" and item["species_id"] == proposal["species_id"] and _iou(item, proposal["box"]) >= 0.7
                        for item in existing
                    )
                    if duplicate:
                        continue
                    self.db.add_annotation(
                        video_id=self.video["id"],
                        frame_number=frame_number,
                        time_seconds=frame_number / max(0.001, float(self.video["fps"])),
                        species_id=proposal["species_id"],
                        track_id=proposal["track_id"],
                        box=proposal["box"],
                        status="pending",
                        source="tracker",
                        confidence=proposal["confidence"],
                        model_id=proposal["model_id"],
                    )
                    added += 1
                self.progress.emit(round(frame_number / total * 100), f"Tracking frame {frame_number:,} of {total:,}")
            self.completed.emit(added)
        except Exception as exc:
            self.failed.emit(str(exc))


class PlaybackWorker(QThread):
    """Decode video without blocking the Qt event loop."""

    frame_ready = Signal(int, object)
    failed = Signal(str)

    def __init__(
        self,
        video: dict[str, Any],
        start_frame: int,
        speed: float,
    ):
        super().__init__()
        self.video = video
        self.start_frame = int(start_frame)
        self._stop_event = threading.Event()
        self._speed_lock = threading.Lock()
        self._speed = max(0.1, float(speed))
        self.last_frame = self.start_frame
        self.last_image: Any | None = None
        self.error_message = ""

    def stop(self) -> None:
        self._stop_event.set()

    def set_speed(self, speed: float) -> None:
        with self._speed_lock:
            self._speed = max(0.1, float(speed))

    def speed(self) -> float:
        with self._speed_lock:
            return self._speed

    def run(self) -> None:
        capture = cv2.VideoCapture(self.video["path"])
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the video for playback")
            frame_number = self.start_frame + 1
            frame_count = max(0, int(self.video["frame_count"]))
            fps = max(0.001, float(self.video["fps"]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            last_emitted_at = 0.0
            frame_credit = 0.0
            while frame_number < frame_count and not self._stop_event.is_set():
                iteration_started = perf_counter()
                ok, image = capture.read()
                if not ok:
                    break
                self.last_frame = frame_number
                self.last_image = image

                now = perf_counter()
                if now - last_emitted_at >= 1 / 30 or frame_number == frame_count - 1:
                    self.frame_ready.emit(frame_number, image)
                    last_emitted_at = now

                speed = self.speed()
                if speed < 1:
                    step = 1
                    target_seconds = 1 / (fps * speed)
                else:
                    frame_credit += speed
                    step = max(1, int(frame_credit))
                    frame_credit -= step
                    target_seconds = 1 / fps
                if frame_number >= frame_count - 1:
                    break
                next_frame = min(frame_count - 1, frame_number + step)
                for _ in range(max(0, next_frame - frame_number - 1)):
                    if not capture.grab():
                        next_frame = frame_count
                        break
                frame_number = next_frame
                remaining = target_seconds - (perf_counter() - iteration_started)
                if remaining > 0:
                    self._stop_event.wait(remaining)
        except Exception as exc:
            self.error_message = str(exc)
            self.failed.emit(self.error_message)
        finally:
            capture.release()


class FrameSeekWorker(QThread):
    """Decode only the newest requested scrub frame away from the UI thread."""

    frame_ready = Signal(int, object)
    failed = Signal(str)

    def __init__(self, video: dict[str, Any]):
        super().__init__()
        self.video = dict(video)
        self._condition = threading.Condition()
        self._requested_frame: int | None = None
        self._stopping = False

    def request_frame(self, frame_number: int) -> None:
        with self._condition:
            self._requested_frame = int(frame_number)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def run(self) -> None:
        capture = cv2.VideoCapture(self.video["path"])
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the video for seeking")
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopping
                        or self._requested_frame is not None
                    )
                    if self._stopping:
                        return
                    frame_number = int(self._requested_frame)
                    self._requested_frame = None
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, image = capture.read()
                if not ok:
                    self.failed.emit(
                        f"Could not decode frame {frame_number:,}"
                    )
                    continue
                with self._condition:
                    if self._stopping:
                        return
                    newer_request_waiting = self._requested_frame is not None
                if not newer_request_waiting:
                    self.frame_ready.emit(frame_number, image)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            capture.release()


class SamMaskWorker(QThread):
    completed = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        engine: SamAssistEngine,
        image: Any,
        points: list[tuple[float, float]],
        labels: list[int],
        revision: int,
    ):
        super().__init__()
        self.engine = engine
        self.image = image.copy()
        self.points = list(points)
        self.labels = list(labels)
        self.revision = revision

    def run(self) -> None:
        try:
            result = self.engine.segment(self.image, self.points, self.labels)
            self.completed.emit(self.revision, result)
        except Exception as exc:
            self.failed.emit(self.revision, str(exc))


class MainWindow(QMainWindow):
    def __init__(self, db: Database, data_dir: Path, *, show_startup_prompt: bool = True):
        super().__init__()
        self.db = db
        self.data_dir = data_dir
        self.training = TrainingCoordinator(db, data_dir)
        self.inference = InferenceEngine(db)
        self.current_project: dict[str, Any] | None = None
        self.current_video: dict[str, Any] | None = None
        self.capture: cv2.VideoCapture | None = None
        self.current_image: Any | None = None
        self.current_frame = 0
        self.selected_annotation_id: str | None = None
        self._annotation_editor_dirty = False
        self._loading_annotation_editor = False
        self.annotation_autosave_timer = QTimer(self)
        self.annotation_autosave_timer.setSingleShot(True)
        self.annotation_autosave_timer.setInterval(300)
        self.annotation_autosave_timer.timeout.connect(self.autosave_annotation_changes)
        self.review_segment_start: int | None = None
        self.tracking_worker: TrackingWorker | None = None
        self.playback_worker: PlaybackWorker | None = None
        self.seek_worker: FrameSeekWorker | None = None
        self._timeline_scrubbing = False
        self._pending_seek_frame: int | None = None
        self._resume_playback_after_scrub = False
        self.sam_engine = SamAssistEngine(data_dir)
        self.sam_capability = self.sam_engine.capability()
        self.sam_worker: SamMaskWorker | None = None
        self.sam_points: list[tuple[float, float, int]] = []
        self.sam_result: SamMaskResult | None = None
        self.sam_revision = 0
        self._sam_rerun_requested = False
        self.sam_context: tuple[str, int] | None = None
        self.sam_manual_override = False
        self.media_focus_mode = False
        self._normal_splitter_sizes: list[int] = []
        self.setWindowTitle("FinFrame — MaxN video annotation")
        self.resize(1480, 920)
        self._build_ui()
        self._apply_style()
        self.refresh_projects()
        self.training_timer = QTimer(self)
        self.training_timer.timeout.connect(self.refresh_training_status)
        self.refresh_training_status()
        self.training_timer.start(1000)
        if show_startup_prompt:
            QTimer.singleShot(0, self.choose_startup_task)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Project")
        self.project_toolbar = toolbar
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        project_label = QLabel("Project")
        project_label.setObjectName("toolbarSectionLabel")
        toolbar.addWidget(project_label)
        self.project_combo = QComboBox()
        self.project_combo.setObjectName("toolbarSelector")
        self.project_combo.setMinimumWidth(260)
        self.project_combo.currentIndexChanged.connect(self.project_changed)
        toolbar.addWidget(self.project_combo)
        new_project = QAction("New project", self)
        new_project.triggered.connect(self.create_project)
        toolbar.addAction(new_project)
        edit_project = QAction("Edit project", self)
        edit_project.triggered.connect(self.edit_project)
        toolbar.addAction(edit_project)
        toolbar.addSeparator()
        media_label = QLabel("Media")
        media_label.setObjectName("toolbarSectionLabel")
        toolbar.addWidget(media_label)
        self.video_combo = QComboBox()
        self.video_combo.setObjectName("toolbarSelector")
        self.video_combo.setMinimumWidth(300)
        self.video_combo.currentIndexChanged.connect(self.video_changed)
        toolbar.addWidget(self.video_combo)
        add_video = QAction("Add video", self)
        add_video.triggered.connect(self.add_video)
        toolbar.addAction(add_video)
        add_images = QAction("Add images / folder", self)
        add_images.triggered.connect(self.choose_image_source)
        toolbar.addAction(add_images)
        backup = QAction("Backup project", self)
        backup.triggered.connect(self.backup_project)
        toolbar.addAction(backup)
        contribution = QAction("Export contribution", self)
        contribution.triggered.connect(self.export_contribution)
        toolbar.addAction(contribution)
        restore = QAction("Import project", self)
        restore.triggered.connect(self.import_project)
        toolbar.addAction(restore)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        workspace_splitter = QSplitter(Qt.Orientation.Vertical)
        self.workspace_splitter = workspace_splitter
        workspace_splitter.setObjectName("workspaceSplitter")
        workspace_splitter.setHandleWidth(8)
        workspace_splitter.setOpaqueResize(False)
        outer.addWidget(workspace_splitter, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        splitter.setObjectName("paneSplitter")
        splitter.setHandleWidth(8)
        splitter.setOpaqueResize(False)
        workspace_splitter.addWidget(splitter)

        species_panel = QGroupBox("Species taxonomy")
        self.species_panel = species_panel
        species_layout = QVBoxLayout(species_panel)
        self.species_search = QLineEdit()
        self.species_search.setPlaceholderText("Search common, scientific or code")
        self.species_search.textChanged.connect(self.refresh_species)
        self.species_list = QListWidget()
        self.species_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.species_list.currentItemChanged.connect(self.active_species_changed)
        self.species_list.itemDoubleClicked.connect(self.edit_species)
        species_actions = QHBoxLayout()
        self.add_species_button = QPushButton("Add species")
        self.add_species_button.setObjectName("addSpeciesButton")
        self.add_species_button.clicked.connect(self.add_species)
        self.edit_species_button = QPushButton("Edit selected")
        self.edit_species_button.setObjectName("editSpeciesButton")
        self.edit_species_button.setEnabled(False)
        self.edit_species_button.clicked.connect(self.edit_species)
        species_actions.addWidget(self.add_species_button)
        species_actions.addWidget(self.edit_species_button)
        species_layout.addWidget(self.species_search)
        species_layout.addWidget(self.species_list, 1)
        species_layout.addLayout(species_actions)
        splitter.addWidget(species_panel)

        video_panel = QWidget()
        self.video_panel = video_panel
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.boxCreated.connect(self.create_manual_box)
        self.canvas.boxChanged.connect(self.canvas_box_changed)
        self.canvas.selectionChanged.connect(self.select_annotation_by_id)
        self.canvas.samPointCreated.connect(self.sam_point_added)
        video_layout.addWidget(self.canvas, 1)
        self.focus_species_panel = QFrame(self.canvas)
        self.focus_species_panel.setObjectName("focusSpeciesPanel")
        self.focus_species_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        focus_species_layout = QVBoxLayout(self.focus_species_panel)
        focus_species_layout.setContentsMargins(10, 10, 10, 10)
        focus_species_layout.setSpacing(6)
        focus_species_title = QLabel("Next fish species")
        focus_species_title.setObjectName("focusSpeciesTitle")
        self.focus_species_current = QLabel("Select a species")
        self.focus_species_current.setObjectName("focusSpeciesCurrent")
        self.focus_species_current.setWordWrap(True)
        self.focus_species_search = QLineEdit()
        self.focus_species_search.setPlaceholderText("Search species or code")
        self.focus_species_search.setClearButtonEnabled(True)
        self.focus_species_search.textChanged.connect(self.refresh_focus_species)
        self.focus_species_list = QListWidget()
        self.focus_species_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.focus_species_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.focus_species_list.setWordWrap(True)
        self.focus_species_list.currentItemChanged.connect(
            self.focus_species_changed
        )
        focus_species_layout.addWidget(focus_species_title)
        focus_species_layout.addWidget(self.focus_species_current)
        focus_species_layout.addWidget(self.focus_species_search)
        focus_species_layout.addWidget(self.focus_species_list, 1)
        self.focus_species_panel.hide()
        self.media_navigation_panel = QWidget()
        self.media_navigation_panel.setObjectName("mediaNavigationPanel")
        navigation_layout = QVBoxLayout(self.media_navigation_panel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(4)
        self.timeline_row = QWidget()
        self.timeline_row.setObjectName("timelineRow")
        self.timeline_row.setMinimumHeight(30)
        timeline_layout = QHBoxLayout(self.timeline_row)
        timeline_layout.setContentsMargins(8, 3, 8, 3)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.sliderPressed.connect(self.timeline_scrub_started)
        self.timeline.sliderMoved.connect(self.timeline_scrub_moved)
        self.timeline.sliderReleased.connect(self.timeline_scrub_finished)
        timeline_layout.addWidget(self.timeline, 1)
        navigation_layout.addWidget(self.timeline_row)
        self.playback_controls_row = QWidget()
        self.playback_controls_row.setObjectName("playbackControlsRow")
        controls = QHBoxLayout(self.playback_controls_row)
        controls.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton("▶")
        self.play_button.clicked.connect(self.toggle_playback)
        self.previous_button = QPushButton("Previous")
        self.previous_button.setIcon(navigation_arrow_icon(-1))
        self.previous_button.setIconSize(QSize(15, 15))
        self.previous_button.clicked.connect(lambda: self.navigate_relative(-1))
        self.following_button = QPushButton("Next")
        self.following_button.setIcon(navigation_arrow_icon(1))
        self.following_button.setIconSize(QSize(15, 15))
        self.following_button.clicked.connect(lambda: self.navigate_relative(1))
        self.back_button = QPushButton("−5 s")
        self.back_button.clicked.connect(lambda: self.seek_video_relative(-round(self._fps() * 5)))
        self.forward_button = QPushButton("+5 s")
        self.forward_button.clicked.connect(lambda: self.seek_video_relative(round(self._fps() * 5)))
        for widget in (self.play_button, self.previous_button, self.following_button, self.back_button, self.forward_button):
            controls.addWidget(widget)
        self.speed_label = QLabel("Speed")
        controls.addWidget(self.speed_label)
        self.playback_speed = QComboBox()
        for speed in (0.5, 1, 1.5, 2, 3, 4, 5, 6):
            self.playback_speed.addItem(f"{speed:g}×", speed)
        self.playback_speed.setCurrentIndex(self.playback_speed.findData(1))
        self.playback_speed.currentIndexChanged.connect(self.playback_speed_changed)
        controls.addWidget(self.playback_speed)
        controls.addStretch(1)
        self.frame_label = QLabel("No media")
        self.frame_label.setObjectName("mediaPositionLabel")
        controls.addWidget(self.frame_label)
        navigation_layout.addWidget(self.playback_controls_row)
        video_layout.addWidget(
            self.media_navigation_panel,
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        self.canvas.imageRectChanged.connect(self.align_media_navigation)
        self.canvas.imageRectChanged.connect(self.position_focus_species_panel)
        sam_controls = QHBoxLayout()
        self.sam_checkbox = QCheckBox("Enable SAM-assisted click annotation")
        self.sam_checkbox.setObjectName("samAssistCheckbox")
        self.sam_checkbox.setChecked(False)
        self.sam_checkbox.setToolTip(
            "Off by default. Click a fish to add a positive point; Shift-click or "
            "right-click unwanted regions to add negative correction points."
        )
        self.sam_checkbox.toggled.connect(self.sam_assist_toggled)
        self.sam_undo_button = QPushButton("Undo point")
        self.sam_undo_button.clicked.connect(self.undo_sam_point)
        self.sam_reset_button = QPushButton("Reset mask")
        self.sam_reset_button.clicked.connect(self.reset_sam_preview)
        self.sam_manual_button = QPushButton("Use one manual box")
        self.sam_manual_button.setToolTip(
            "Temporarily draw one bounding box, then return automatically to SAM point mode"
        )
        self.sam_manual_button.clicked.connect(self.use_manual_box_mode)
        self.sam_accept_button = QPushButton("Accept mask + box")
        self.sam_accept_button.setObjectName("samAcceptButton")
        self.sam_accept_button.clicked.connect(self.accept_sam_mask)
        for widget in (
            self.sam_checkbox,
            self.sam_undo_button,
            self.sam_reset_button,
            self.sam_manual_button,
            self.sam_accept_button,
        ):
            sam_controls.addWidget(widget)
        video_layout.addLayout(sam_controls)
        self.sam_status = QLabel(self.sam_capability.message)
        self.sam_status.setWordWrap(True)
        video_layout.addWidget(self.sam_status)
        ai_controls = QHBoxLayout()
        self.ai_frame_button = QPushButton("AI suggest current frame")
        self.ai_frame_button.clicked.connect(self.suggest_current_frame)
        self.byte_button = QPushButton("Track entire video · ByteTrack")
        self.byte_button.clicked.connect(lambda: self.start_tracking("bytetrack"))
        self.bot_button = QPushButton("Track entire video · BoT-SORT")
        self.bot_button.clicked.connect(lambda: self.start_tracking("botsort"))
        self.tracking_progress = QProgressBar()
        self.tracking_progress.setRange(0, 100)
        self.tracking_progress.setValue(0)
        for widget in (self.ai_frame_button, self.byte_button, self.bot_button, self.tracking_progress):
            ai_controls.addWidget(widget)
        video_layout.addLayout(ai_controls)
        splitter.addWidget(video_panel)

        self.annotation_panel = QGroupBox("Frame annotations")
        self.annotation_panel.setMinimumWidth(400)
        annotation_outer = QVBoxLayout(self.annotation_panel)
        annotation_outer.setContentsMargins(6, 8, 6, 6)
        annotation_scroll = QScrollArea()
        annotation_scroll.setWidgetResizable(True)
        annotation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        annotation_scroll.setObjectName("annotationScroll")
        annotation_content = QWidget()
        annotation_layout = QVBoxLayout(annotation_content)
        annotation_layout.setContentsMargins(4, 4, 4, 4)
        annotation_scroll.setWidget(annotation_content)
        annotation_outer.addWidget(annotation_scroll)
        self.frame_counts = QLabel("Verified fish: 0")
        self.frame_complete = QCheckBox("Frame complete — all visible fish are boxed")
        self.frame_complete.toggled.connect(self.frame_complete_toggled)
        self.training_keyframe_status = QLabel("Not complete; excluded from final MaxN and training")
        self.annotation_editor_status = QLabel("Select a species on the left before drawing")
        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(["Status", "Species", "Track", "Source", "Confidence"])
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.annotation_table.itemSelectionChanged.connect(self.annotation_selected)
        self.annotation_table.setMinimumHeight(185)
        annotation_header = self.annotation_table.horizontalHeader()
        annotation_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        annotation_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        annotation_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        annotation_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        annotation_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.annotation_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        annotation_layout.addWidget(self.frame_counts)
        annotation_layout.addWidget(self.frame_complete)
        annotation_layout.addWidget(self.training_keyframe_status)
        annotation_layout.addWidget(self.annotation_table, 1)
        annotation_layout.addWidget(self.annotation_editor_status)
        form = QFormLayout()
        self.annotation_species = QComboBox()
        self.annotation_track = QLineEdit()
        self.annotation_stage = QComboBox()
        self.annotation_stage.addItems(["Adult", "Juvenile", "Unknown"])
        self.annotation_activity = QComboBox()
        self.annotation_activity.addItems(["Passing", "Feeding", "Schooling", "Resting", "Unknown"])
        self.annotation_uncertain = QCheckBox("Uncertain")
        self.annotation_species.currentIndexChanged.connect(self.annotation_editor_changed)
        self.annotation_track.textEdited.connect(self.annotation_editor_changed)
        self.annotation_track.editingFinished.connect(self.autosave_annotation_changes)
        self.annotation_stage.currentIndexChanged.connect(self.annotation_editor_changed)
        self.annotation_activity.currentIndexChanged.connect(self.annotation_editor_changed)
        self.annotation_uncertain.toggled.connect(self.annotation_editor_changed)
        form.addRow("Species", self.annotation_species)
        form.addRow("Track ID", self.annotation_track)
        form.addRow("Life stage", self.annotation_stage)
        form.addRow("Activity", self.annotation_activity)
        form.addRow("", self.annotation_uncertain)
        annotation_layout.addLayout(form)
        action_row = QGridLayout()
        self.approve_button = QPushButton("Approve proposal")
        self.approve_button.clicked.connect(self.approve_annotation)
        self.approve_button.setEnabled(False)
        self.reject_button = QPushButton("Reject + stop track")
        self.reject_button.clicked.connect(self.reject_annotation)
        self.reject_button.setToolTip("Reject this proposal and stop propagating its track")
        self.reject_button.setEnabled(False)
        self.approve_frame_button = QPushButton("Approve all unchanged proposals on frame")
        self.approve_frame_button.clicked.connect(self.approve_frame_proposals)
        next_pending_button = QPushButton("Next pending frame")
        next_pending_button.clicked.connect(self.go_to_next_pending_frame)
        self.approve_segment_button = QPushButton("Approve watched segment")
        self.approve_segment_button.clicked.connect(self.approve_watched_segment)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_annotation)
        self.clear_video_boxes_button = QPushButton("Clear all boxes from this media")
        self.clear_video_boxes_button.setObjectName("dangerButton")
        self.clear_video_boxes_button.setToolTip(
            "Permanently delete every bounding box from the selected image or video"
        )
        self.clear_video_boxes_button.setEnabled(False)
        self.clear_video_boxes_button.clicked.connect(self.clear_all_video_boxes)
        action_row.addWidget(self.approve_button, 0, 0)
        action_row.addWidget(self.reject_button, 0, 1)
        action_row.addWidget(self.approve_frame_button, 1, 0, 1, 2)
        action_row.addWidget(next_pending_button, 2, 0, 1, 2)
        action_row.addWidget(self.approve_segment_button, 3, 0, 1, 2)
        action_row.addWidget(delete_button, 4, 0, 1, 2)
        action_row.addWidget(self.clear_video_boxes_button, 5, 0, 1, 2)
        annotation_layout.addLayout(action_row)
        annotation_layout.addStretch(1)
        splitter.addWidget(self.annotation_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, True)
        splitter.setSizes([230, 830, 420])
        splitter.handle(1).setToolTip("Drag to resize the species and media panels")
        splitter.handle(2).setToolTip("Drag to resize the media and annotation panels")

        tabs = QTabWidget()
        self.bottom_tabs = tabs
        maxn_tab = QWidget()
        maxn_layout = QVBoxLayout(maxn_tab)
        maxn_layout.setContentsMargins(8, 8, 8, 8)
        self.maxn_status = QLabel(
            "Live MaxN updates from verified boxes; Final MaxN includes completed frames only."
        )
        self.maxn_status.setWordWrap(True)
        self.maxn_table = QTableWidget(0, 6)
        self.maxn_table.setHorizontalHeaderLabels(
            ["Species", "Code", "Live MaxN", "Final MaxN", "Live peak frame", "Live peak time"]
        )
        self.maxn_table.horizontalHeader().setStretchLastSection(True)
        maxn_layout.addWidget(self.maxn_status)
        maxn_layout.addWidget(self.maxn_table, 1)
        tabs.addTab(maxn_tab, "MaxN summary")
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)
        self.dataset_stats = QLabel("0 verified observation boxes · 0 complete frames · 0 selected training keyframes")
        export_row = QHBoxLayout()
        export_coco = QPushButton("Export completed observations · COCO")
        export_coco.clicked.connect(lambda: self.export_training_data("coco"))
        export_yolo = QPushButton("Export completed observations · YOLO")
        export_yolo.clicked.connect(lambda: self.export_training_data("yolo"))
        export_row.addWidget(export_coco)
        export_row.addWidget(export_yolo)
        dataset_layout.addWidget(self.dataset_stats)
        dataset_layout.addLayout(export_row)
        tabs.addTab(dataset_tab, "Dataset")
        training_tab = QWidget()
        training_layout = QGridLayout(training_tab)
        self.training_summary = QLabel("Training idle")
        self.training_summary.setWordWrap(True)
        self.active_model_label = QLabel("Active model: none")
        self.training_progress = QProgressBar()
        self.train_now_button = QPushButton("Train now from selected keyframes")
        self.train_now_button.setToolTip(
            "Training starts only when this button is pressed and enough reviewed data is available"
        )
        self.train_now_button.clicked.connect(
            lambda: self.training.request_training(reason="student requested")
        )
        training_layout.addWidget(self.training_summary, 0, 0, 1, 3)
        training_layout.addWidget(self.active_model_label, 1, 0, 1, 3)
        training_layout.addWidget(self.train_now_button, 2, 0, 1, 3)
        training_layout.addWidget(self.training_progress, 3, 0, 1, 3)
        tabs.addTab(training_tab, "AI training")
        workspace_splitter.addWidget(tabs)
        workspace_splitter.setStretchFactor(0, 1)
        workspace_splitter.setStretchFactor(1, 0)
        workspace_splitter.setCollapsible(0, False)
        workspace_splitter.setCollapsible(1, True)
        workspace_splitter.setSizes([665, 225])
        workspace_splitter.handle(1).setToolTip(
            "Drag to resize or collapse the MaxN, dataset and training panel"
        )
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        delete_shortcut = QAction(self)
        delete_shortcut.setShortcut(QKeySequence.StandardKey.Delete)
        delete_shortcut.triggered.connect(self.delete_annotation)
        self.addAction(delete_shortcut)
        focus_shortcut = QAction(self)
        focus_shortcut.setShortcut(QKeySequence("F11"))
        focus_shortcut.triggered.connect(self.toggle_media_focus)
        self.addAction(focus_shortcut)
        restore_shortcut = QAction(self)
        restore_shortcut.setShortcut(QKeySequence("Esc"))
        restore_shortcut.triggered.connect(self.restore_media_layout)
        self.addAction(restore_shortcut)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f3; color: #14251f; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }
            QToolBar { background: #102a22; color: #f4fbf8; spacing: 7px; padding: 6px 8px; border: 0; }
            QToolBar QLabel#toolbarSectionLabel { color: #d7e5df; background: transparent; font-weight: 700; padding: 0 3px; }
            QToolBar QComboBox#toolbarSelector { background: #f8fbf9; color: #14251f; border: 1px solid #86a79a; border-radius: 5px; padding: 5px 26px 5px 8px; }
            QToolBar QComboBox#toolbarSelector:hover { border-color: #70c3a6; background: white; }
            QToolBar QComboBox#toolbarSelector:focus { border: 2px solid #41a88a; padding: 4px 25px 4px 7px; }
            QToolBar QComboBox#toolbarSelector QAbstractItemView { background: white; color: #14251f; selection-background-color: #d6e7df; selection-color: #14251f; }
            QToolBar QToolButton { background: transparent; color: #eef8f4; border: 1px solid transparent; border-radius: 5px; padding: 6px 8px; }
            QToolBar QToolButton:hover { background: #1d493c; border-color: #467466; }
            QToolBar QToolButton:pressed { background: #28614f; }
            QToolBar::separator { background: #41665a; width: 1px; margin: 4px 3px; }
            QFrame#focusSpeciesPanel { background: rgba(10, 31, 25, 232); border: 1px solid #6b9586; border-radius: 8px; }
            QFrame#focusSpeciesPanel QLabel#focusSpeciesTitle { color: #f4fbf8; background: transparent; font-size: 14px; font-weight: 700; }
            QFrame#focusSpeciesPanel QLabel#focusSpeciesCurrent { color: #bfe7d8; background: transparent; font-weight: 600; }
            QFrame#focusSpeciesPanel QLineEdit { background: #f8fbf9; color: #14251f; }
            QFrame#focusSpeciesPanel QListWidget { background: rgba(248, 251, 249, 245); color: #14251f; border-color: #6b9586; }
            QFrame#focusSpeciesPanel QListWidget::item { padding: 5px; }
            QFrame#focusSpeciesPanel QListWidget::item:selected { background: #b9dfd1; color: #102a22; }
            QWidget#timelineRow { background: #e7eeea; border: 1px solid #c5d3cd; border-radius: 5px; }
            QLabel#mediaPositionLabel { color: #344b42; background: transparent; font-weight: 600; padding: 0 4px; }
            QGroupBox { font-weight: 700; border: 1px solid #cbd8d2; border-radius: 8px; margin-top: 10px; padding-top: 12px; background: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #e5eee9; border: 1px solid #b7cbc2; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { background: #d6e7df; }
            QPushButton:disabled { color: #82918b; background: #edf1ef; }
            QPushButton#dangerButton { color: #8f2f25; background: #fff2ef; border-color: #dca69f; }
            QPushButton#dangerButton:hover { background: #ffe4de; }
            QLineEdit, QComboBox, QTextEdit { background: white; border: 1px solid #b8c8c1; border-radius: 5px; padding: 5px; }
            QTableWidget, QListWidget { background: white; border: 1px solid #cbd8d2; alternate-background-color: #f4f8f6; }
            QScrollArea#annotationScroll { border: 0; background: transparent; }
            QHeaderView::section { background: #e8efeb; padding: 5px; border: 0; border-bottom: 1px solid #bdccc5; }
            QSplitter::handle { background: #d2dfd9; border-radius: 3px; }
            QSplitter::handle:hover { background: #70c3a6; }
            QSplitter::handle:pressed { background: #41a88a; }
            QTabWidget::pane { border: 1px solid #cbd8d2; background: white; }
            QTabBar::tab { padding: 7px 16px; background: #e4ece8; }
            QTabBar::tab:selected { background: white; font-weight: 700; }
            QProgressBar { border: 1px solid #b7cbc2; border-radius: 5px; text-align: center; background: white; }
            QProgressBar::chunk { background: #41a88a; border-radius: 4px; }
        """)

    def closeEvent(self, event: Any) -> None:
        if not self._persist_selected_annotation():
            event.ignore()
            return
        self.stop_playback(refresh=False)
        self._stop_seek_worker()
        if self.sam_worker and self.sam_worker.isRunning():
            self.sam_worker.wait()
        if hasattr(self, "training_timer"):
            self.training_timer.stop()
        if self.capture:
            self.capture.release()
        event.accept()

    def _fps(self) -> float:
        return float(self.current_video["fps"]) if self.current_video else 25.0

    @staticmethod
    def _timecode(seconds: float) -> str:
        minutes, second = divmod(max(0.0, seconds), 60)
        hours, minutes = divmod(int(minutes), 60)
        return f"{hours:02d}:{minutes:02d}:{second:06.3f}" if hours else f"{minutes:02d}:{second:06.3f}"

    def choose_startup_task(self) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Start annotating")
        dialog.setText("What would you like to annotate?")
        dialog.setInformativeText(
            "Completed images and diverse completed video keyframes contribute boxes to the same shared training dataset."
        )
        video_button = dialog.addButton("Annotate video", QMessageBox.ButtonRole.ActionRole)
        image_button = dialog.addButton("Annotate images", QMessageBox.ButtonRole.ActionRole)
        dialog.addButton("Open existing project", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        selected = dialog.clickedButton()
        if selected not in {video_button, image_button}:
            return
        if not self._ensure_current_project():
            return
        if selected is image_button:
            self.choose_image_source()
        else:
            self.add_video()

    def _ensure_current_project(self) -> bool:
        if self.current_project:
            return True
        projects = self.db.list_projects()
        if projects:
            choices = ["Create a new project", *(project["name"] for project in projects)]
            choice, accepted = QInputDialog.getItem(self, "Choose project", "Add media to", choices, 0, False)
            if not accepted:
                return False
            if choice == choices[0]:
                self.create_project()
            else:
                selected = projects[choices.index(choice) - 1]
                self.refresh_projects(selected["id"])
        else:
            self.create_project()
        return self.current_project is not None

    def refresh_projects(self, select_id: str | None = None) -> None:
        projects = self.db.list_projects()
        with QSignalBlocker(self.project_combo):
            self.project_combo.clear()
            self.project_combo.addItem("Select a project…", None)
            for project in projects:
                self.project_combo.addItem(project["name"], project["id"])
            if select_id:
                index = self.project_combo.findData(select_id)
                self.project_combo.setCurrentIndex(max(0, index))
        if select_id:
            self.project_changed(self.project_combo.currentIndex())

    def create_project(self) -> None:
        dialog = ProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if not values["name"]:
            QMessageBox.warning(self, "Project name required", "Enter a project name.")
            return
        project = self.db.create_project(values.pop("name"), **values)
        self.refresh_projects(project["id"])

    def edit_project(self) -> None:
        if not self.current_project:
            return
        dialog = ProjectDialog(self, self.current_project)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if not values["name"]:
            QMessageBox.warning(self, "Project name required", "Enter a project name.")
            return
        self.db.update_project(self.current_project["id"], **values)
        self.refresh_projects(self.current_project["id"])

    def project_changed(self, index: int) -> None:
        if not self._persist_selected_annotation():
            return
        self.stop_playback(refresh=False)
        self._stop_seek_worker()
        project_id = self.project_combo.itemData(index)
        self.current_project = self.db.get_project(project_id) if project_id else None
        self.refresh_videos()
        self.refresh_species()

    def refresh_videos(self, select_id: str | None = None) -> None:
        videos = self.db.list_videos(self.current_project["id"]) if self.current_project else []
        with QSignalBlocker(self.video_combo):
            self.video_combo.clear()
            self.video_combo.addItem("Select media…", None)
            for video in videos:
                kind = "Image" if video.get("media_type") == "image" else "Video"
                self.video_combo.addItem(f"[{kind}] {video['file_name']}", video["id"])
                self.video_combo.setItemData(
                    self.video_combo.count() - 1,
                    video.get("media_type", "video"),
                    Qt.ItemDataRole.UserRole + 1,
                )
            if select_id:
                self.video_combo.setCurrentIndex(max(0, self.video_combo.findData(select_id)))
        if select_id:
            self.video_changed(self.video_combo.currentIndex())

    def add_video(self) -> None:
        if not self.current_project:
            QMessageBox.information(self, "Create a project", "Create or select a survey project before adding a video.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open survey video", "", "Video files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)")
        if not path:
            return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            QMessageBox.critical(self, "Video error", "OpenCV could not read this video.")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        video = self.db.add_video(self.current_project["id"], path, duration=frames / max(0.001, fps), width=width, height=height, fps=fps, frame_count=frames)
        self.refresh_videos(video["id"])

    def choose_image_source(self) -> None:
        if not self.current_project:
            QMessageBox.information(self, "Create a project", "Create or select a survey project before adding images.")
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Add survey images")
        dialog.setText("Would you like to choose image files or import a folder?")
        dialog.setInformativeText("Folder import includes supported images in all subfolders.")
        files_button = dialog.addButton("Choose image files", QMessageBox.ButtonRole.ActionRole)
        folder_button = dialog.addButton("Choose a folder", QMessageBox.ButtonRole.ActionRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        selected = dialog.clickedButton()
        if selected is files_button:
            self.add_images()
        elif selected is folder_button:
            self.add_image_folder()

    def add_images(self) -> None:
        if not self.current_project:
            QMessageBox.information(self, "Create a project", "Create or select a survey project before adding images.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open survey images",
            "",
            IMAGE_FILE_FILTER,
        )
        if not paths:
            return
        self._import_images(paths)

    def add_image_folder(self) -> None:
        if not self.current_project:
            QMessageBox.information(self, "Create a project", "Create or select a survey project before adding images.")
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose a folder containing survey images",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        paths = discover_image_files(folder)
        if not paths:
            QMessageBox.information(
                self,
                "No supported images found",
                "The selected folder and its subfolders contain no JPG, PNG, TIFF, BMP or WebP images.",
            )
            return
        if QMessageBox.question(
            self,
            "Import image folder",
            f"Import {len(paths):,} supported image{'s' if len(paths) != 1 else ''} from this folder and its subfolders?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._import_images(paths)

    def _import_images(self, paths: Sequence[str | Path]) -> None:
        if not self.current_project:
            return
        first_id = None
        unreadable = []
        for path in paths:
            image_path = Path(path).expanduser().resolve()
            image = cv2.imread(str(image_path))
            if image is None:
                unreadable.append(image_path.name)
                continue
            height, width = image.shape[:2]
            media = self.db.add_video(
                self.current_project["id"],
                image_path,
                duration=0,
                width=width,
                height=height,
                fps=1,
                frame_count=1,
                media_type="image",
            )
            self.db.ensure_frame(media["id"], 0, 0)
            self.db.update_frame(media["id"], 0, image_path=image_path)
            if first_id is None:
                first_id = media["id"]
        if first_id:
            self.refresh_videos(first_id)
            self.statusBar().showMessage(
                f"Added {len(paths) - len(unreadable):,} image{'s' if len(paths) - len(unreadable) != 1 else ''}",
                5000,
            )
        if unreadable:
            QMessageBox.warning(self, "Some images were skipped", "OpenCV could not read:\n" + "\n".join(unreadable))

    def video_changed(self, index: int) -> None:
        if not self._persist_selected_annotation():
            return
        self.stop_playback(refresh=False)
        self._stop_seek_worker()
        video_id = self.video_combo.itemData(index)
        if not video_id:
            return
        video = self.db.get_video(video_id)
        stored_frames = [
            frame for frame in self.db.frames_for_video(video_id)
            if frame.get("image_path") and Path(frame["image_path"]).is_file()
        ]
        source_available = Path(video["path"]).is_file()
        if not source_available and not stored_frames:
            media_name = "image" if video.get("media_type") == "image" else "video"
            if QMessageBox.question(
                self,
                f"Source {media_name} missing",
                f"The source {media_name} is unavailable at:\n{video['path']}\n\nRelink it now?",
            ) != QMessageBox.StandardButton.Yes:
                return
            file_filter = (
                "Image files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)"
                if media_name == "image"
                else "Video files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)"
            )
            replacement, _ = QFileDialog.getOpenFileName(self, f"Relink source {media_name}", "", file_filter)
            if not replacement:
                return
            if media_name == "image":
                image = cv2.imread(replacement)
                if image is None:
                    QMessageBox.warning(self, "Image error", "OpenCV could not read the replacement image.")
                    return
                height, width = image.shape[:2]
                video = self.db.relink_video(video_id, replacement, duration=0, width=width, height=height, fps=1, frame_count=1)
            else:
                metadata = cv2.VideoCapture(replacement)
                if not metadata.isOpened():
                    QMessageBox.warning(self, "Video error", "OpenCV could not read the replacement video.")
                    return
                fps = float(metadata.get(cv2.CAP_PROP_FPS) or video["fps"])
                frames = int(metadata.get(cv2.CAP_PROP_FRAME_COUNT) or video["frame_count"])
                width = int(metadata.get(cv2.CAP_PROP_FRAME_WIDTH) or video["width"])
                height = int(metadata.get(cv2.CAP_PROP_FRAME_HEIGHT) or video["height"])
                metadata.release()
                video = self.db.relink_video(video_id, replacement, duration=frames / max(0.001, fps), width=width, height=height, fps=fps, frame_count=frames)
            source_available = True
        if self.capture:
            self.capture.release()
        self.review_segment_start = None
        self.capture = (
            cv2.VideoCapture(video["path"])
            if video.get("media_type") == "video" and source_available
            else None
        )
        self.current_video = video
        self.timeline.setRange(0, max(0, int(video["frame_count"]) - 1))
        start_frame = int(stored_frames[0]["frame_number"]) if not source_available and stored_frames else 0
        self.seek_frame(start_frame)
        self._configure_media_controls()
        kind = "image" if video.get("media_type") == "image" else "video"
        detail = "still image" if kind == "image" else f"{video['fps']:.2f} fps"
        self.statusBar().showMessage(f"Opened {kind} {video['file_name']} · {video['width']}×{video['height']} · {detail}")

    def align_media_navigation(self, image_rect: Any) -> None:
        if not hasattr(self, "media_navigation_panel"):
            return
        rendered_width = max(
            1,
            min(self.canvas.width(), int(round(float(image_rect.width())))),
        )
        self.media_navigation_panel.setFixedWidth(rendered_width)

    def position_focus_species_panel(self, image_rect: Any) -> None:
        """Keep the focus-mode picker in the left gutter when one is available."""
        if not hasattr(self, "focus_species_panel"):
            return
        margin = 12
        canvas_width = max(1, self.canvas.width())
        canvas_height = max(1, self.canvas.height())
        left_gutter = max(0, int(round(float(image_rect.left()))))
        if left_gutter >= 220 + (margin * 2):
            panel_width = min(250, left_gutter - (margin * 2))
        else:
            panel_width = min(250, max(190, canvas_width // 5))
        panel_height = max(220, min(540, canvas_height - (margin * 2)))
        self.focus_species_panel.setGeometry(
            margin,
            margin,
            panel_width,
            panel_height,
        )
        if self.media_focus_mode:
            self.focus_species_panel.raise_()

    def toggle_media_focus(self, *_args: Any) -> None:
        if not self.media_focus_mode and not self.current_video:
            return
        enabling = not self.media_focus_mode
        if enabling:
            self._normal_splitter_sizes = self.main_splitter.sizes()
        self.media_focus_mode = enabling
        self.project_toolbar.setVisible(not enabling)
        self.species_panel.setVisible(not enabling)
        self.annotation_panel.setVisible(not enabling)
        self.bottom_tabs.setVisible(not enabling)
        self.focus_species_panel.setVisible(enabling)
        if enabling:
            self.refresh_focus_species()
            self.focus_species_panel.raise_()
        if not enabling and self._normal_splitter_sizes:
            sizes = list(self._normal_splitter_sizes)
            QTimer.singleShot(0, lambda: self.main_splitter.setSizes(sizes))
        self._configure_media_controls()
        QTimer.singleShot(
            0,
            lambda: (
                self.align_media_navigation(self.canvas._image_rect()),
                self.position_focus_species_panel(self.canvas._image_rect()),
            ),
        )
        self.statusBar().showMessage(
            "Media enlarged · press Esc or Restore layout to return"
            if enabling
            else "Normal workspace restored",
            5000,
        )

    def restore_media_layout(self, *_args: Any) -> None:
        if self.media_focus_mode:
            self.toggle_media_focus()

    def _configure_media_controls(self) -> None:
        playable = bool(
            self.current_video
            and self.current_video.get("media_type") == "video"
            and self.capture
            and self.capture.isOpened()
        )
        is_image = bool(self.current_video and self.current_video.get("media_type") == "image")
        playing = bool(self.playback_worker and self.playback_worker.isRunning())
        seeking = bool(
            self._timeline_scrubbing or self._pending_seek_frame is not None
        )
        busy = playing or seeking
        has_media = bool(self.current_video)
        for control in (
            self.play_button,
            self.back_button,
            self.forward_button,
            self.byte_button,
            self.bot_button,
        ):
            control.setEnabled(
                playable
                and (
                    (control is self.play_button and not seeking)
                    or (control is not self.play_button and not busy)
                )
            )
        self.timeline.setEnabled(playable)
        self.playback_speed.setEnabled(playable)
        sam_available = bool(
            self.sam_capability.available
            and self.current_video
            and self.current_image is not None
            and not busy
        )
        self.sam_checkbox.setEnabled(sam_available)
        self.approve_segment_button.setEnabled(playable and not busy)
        media_noun = "image" if is_image else "video"
        self.clear_video_boxes_button.setText(f"Clear all boxes from this {media_noun}")
        self.clear_video_boxes_button.setToolTip(
            f"Permanently delete every bounding box from the selected {media_noun}"
        )
        self.clear_video_boxes_button.setEnabled(has_media and not busy)
        self.media_navigation_panel.setVisible(has_media)
        self.timeline_row.setVisible(has_media and not is_image)
        for video_only_control in (
            self.play_button,
            self.back_button,
            self.forward_button,
            self.speed_label,
            self.playback_speed,
        ):
            video_only_control.setVisible(has_media and not is_image)
        previous_description = "Previous image" if is_image else "Previous frame"
        next_description = "Next image" if is_image else "Next frame"
        self.previous_button.setToolTip(previous_description)
        self.previous_button.setAccessibleName(previous_description)
        self.following_button.setToolTip(next_description)
        self.following_button.setAccessibleName(next_description)
        self.previous_button.setEnabled(playable or (is_image and self._adjacent_image_index(-1) is not None))
        self.following_button.setEnabled(playable or (is_image and self._adjacent_image_index(1) is not None))
        self.canvas.setEnabled(not busy)
        self.canvas.set_sam_mode(
            bool(sam_available and self.sam_checkbox.isChecked() and not self.sam_manual_override)
        )
        self.annotation_panel.setEnabled(not busy)
        self._refresh_sam_controls()

    def _adjacent_image_index(self, direction: int) -> int | None:
        index = self.video_combo.currentIndex() + (1 if direction > 0 else -1)
        while 0 <= index < self.video_combo.count():
            media_type = self.video_combo.itemData(index, Qt.ItemDataRole.UserRole + 1)
            if media_type == "image":
                return index
            index += 1 if direction > 0 else -1
        return None

    def _image_position(self) -> tuple[int, int]:
        image_indices = [
            index
            for index in range(self.video_combo.count())
            if self.video_combo.itemData(index, Qt.ItemDataRole.UserRole + 1) == "image"
        ]
        try:
            position = image_indices.index(self.video_combo.currentIndex()) + 1
        except ValueError:
            position = 1 if image_indices else 0
        return position, len(image_indices)

    def navigate_relative(self, direction: int) -> None:
        if not self.current_video:
            return
        if self.current_video.get("media_type") == "image":
            target = self._adjacent_image_index(direction)
            if target is not None:
                self.video_combo.setCurrentIndex(target)
            return
        self.stop_playback(refresh=False)
        self.seek_frame(self.current_frame + direction)

    def seek_video_relative(self, frames: int) -> None:
        self.stop_playback(refresh=False)
        self.seek_frame(self.current_frame + frames)

    def _ensure_seek_worker(self) -> bool:
        if (
            not self.current_video
            or self.current_video.get("media_type") != "video"
        ):
            return False
        if (
            self.seek_worker
            and self.seek_worker.isRunning()
            and self.seek_worker.video.get("id") == self.current_video.get("id")
        ):
            return True
        self._stop_seek_worker(reset_scrub_state=False)
        worker = FrameSeekWorker(self.current_video)
        worker.frame_ready.connect(self.timeline_seek_frame_ready)
        worker.failed.connect(self.timeline_seek_failed)
        self.seek_worker = worker
        worker.start()
        return True

    def _stop_seek_worker(self, *, reset_scrub_state: bool = True) -> None:
        worker = self.seek_worker
        self.seek_worker = None
        if worker and worker.isRunning():
            worker.stop()
            worker.wait()
        if reset_scrub_state:
            self._timeline_scrubbing = False
            self._pending_seek_frame = None
            self._resume_playback_after_scrub = False

    def timeline_scrub_started(self) -> None:
        if (
            not self.current_video
            or self.current_video.get("media_type") != "video"
            or not self._persist_selected_annotation()
        ):
            return
        self._timeline_scrubbing = True
        self._pending_seek_frame = None
        self._resume_playback_after_scrub = bool(
            self.playback_worker and self.playback_worker.isRunning()
        )
        if self._resume_playback_after_scrub:
            self.stop_playback(refresh=False)
        if not self._ensure_seek_worker():
            self._timeline_scrubbing = False
            self._resume_playback_after_scrub = False
            return
        self.canvas.set_annotations([])
        self._configure_media_controls()

    def timeline_scrub_moved(self, frame_number: int) -> None:
        if not self._timeline_scrubbing:
            self.timeline_scrub_started()
        if not self._timeline_scrubbing or not self.current_video:
            return
        frame_number = max(
            0,
            min(
                int(self.current_video["frame_count"]) - 1,
                int(frame_number),
            ),
        )
        if frame_number != self.current_frame:
            self.review_segment_start = None
        self._pending_seek_frame = frame_number
        seconds = frame_number / max(0.001, self._fps())
        duration = float(self.current_video.get("duration") or 0)
        self.frame_label.setText(
            f"Seeking frame {frame_number:,} · "
            f"{self._timecode(seconds)} / {self._timecode(duration)}"
        )
        if self.seek_worker:
            self.seek_worker.request_frame(frame_number)

    def timeline_scrub_finished(self) -> None:
        if not self._timeline_scrubbing or not self.current_video:
            return
        target = max(
            0,
            min(
                int(self.current_video["frame_count"]) - 1,
                int(self.timeline.value()),
            ),
        )
        self._timeline_scrubbing = False
        self._pending_seek_frame = target
        if self.current_frame == target and self.current_image is not None:
            self._complete_timeline_seek()
            return
        if self.seek_worker:
            self.seek_worker.request_frame(target)
        self._configure_media_controls()

    def timeline_seek_frame_ready(self, frame_number: int, image: Any) -> None:
        worker = self.sender()
        if (
            worker is None
            or worker is not self.seek_worker
            or not self.current_video
            or worker.video.get("id") != self.current_video.get("id")
            or frame_number != self._pending_seek_frame
        ):
            return
        self._display_frame(
            frame_number,
            image,
            refresh_maxn=not self._timeline_scrubbing,
            refresh_annotations=not self._timeline_scrubbing,
        )
        if not self._timeline_scrubbing:
            self._complete_timeline_seek(refresh_annotations=False)

    def timeline_seek_failed(self, message: str) -> None:
        worker = self.sender()
        if worker is not self.seek_worker:
            return
        self._pending_seek_frame = None
        self._timeline_scrubbing = False
        self._resume_playback_after_scrub = False
        self._configure_media_controls()
        self.statusBar().showMessage(f"Could not seek video: {message}", 8000)

    def _complete_timeline_seek(
        self,
        *,
        refresh_annotations: bool = True,
    ) -> None:
        target = self.current_frame
        self._pending_seek_frame = None
        with QSignalBlocker(self.timeline):
            self.timeline.setValue(target)
        if refresh_annotations:
            self.refresh_frame_annotations(refresh_maxn=True)
        resume = self._resume_playback_after_scrub
        self._resume_playback_after_scrub = False
        self._configure_media_controls()
        if (
            resume
            and self.current_video
            and self.current_frame
            < int(self.current_video["frame_count"]) - 1
        ):
            self.toggle_playback()

    def seek_frame(self, frame_number: int) -> None:
        if not self.current_video:
            return
        if self.playback_worker and self.playback_worker.isRunning():
            self.stop_playback(refresh=False)
        frame_number = max(0, min(int(self.current_video["frame_count"]) - 1, int(frame_number)))
        previous_frame = self.current_frame
        if frame_number != previous_frame and not self._persist_selected_annotation():
            return
        if self.current_image is not None and frame_number != previous_frame:
            self.review_segment_start = None
        image = None
        if self.current_video.get("media_type") == "image":
            try:
                frame = self.db.get_frame(self.current_video["id"], 0)
            except KeyError:
                frame = {}
            source = frame.get("image_path") if frame.get("image_path") and Path(frame["image_path"]).is_file() else self.current_video["path"]
            image = cv2.imread(source)
        elif self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = self.capture.read()
            if not ok:
                image = None
        else:
            try:
                frame = self.db.get_frame(self.current_video["id"], frame_number)
            except KeyError:
                frame = {}
            if frame.get("image_path") and Path(frame["image_path"]).is_file():
                image = cv2.imread(frame["image_path"])
        if image is None:
            self.stop_playback(refresh=False)
            self.play_button.setText("▶")
            self.statusBar().showMessage("This frame is unavailable; relink the full source video to browse unannotated frames", 8000)
            return
        self._display_frame(frame_number, image, refresh_maxn=True)

    def _display_frame(
        self,
        frame_number: int,
        image: Any,
        *,
        refresh_maxn: bool,
        refresh_annotations: bool = True,
    ) -> None:
        context = (
            (str(self.current_video["id"]), int(frame_number))
            if self.current_video
            else None
        )
        if self.sam_context is not None and self.sam_context != context:
            self.reset_sam_preview()
        self.current_frame = frame_number
        self.current_image = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(qimage)
        if not self._timeline_scrubbing:
            with QSignalBlocker(self.timeline):
                self.timeline.setValue(frame_number)
        if self.current_video and self.current_video.get("media_type") == "image":
            position, image_count = self._image_position()
            self.frame_label.setText(f"Image {position:,} of {image_count:,}")
        else:
            seconds = frame_number / max(0.001, self._fps())
            duration = float(self.current_video.get("duration") or 0) if self.current_video else 0
            self.frame_label.setText(
                f"Frame {frame_number:,} · {self._timecode(seconds)} / {self._timecode(duration)}"
            )
        if refresh_annotations:
            self.refresh_frame_annotations(refresh_maxn=refresh_maxn)

    def toggle_playback(self) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video" or not self.capture:
            return
        if self.playback_worker and self.playback_worker.isRunning():
            self.stop_playback(refresh=True)
            return
        if not self._persist_selected_annotation():
            return
        if self.sam_checkbox.isChecked():
            self.sam_manual_override = False
            self.reset_sam_preview()
            self.canvas.set_sam_mode(False)
            self.sam_status.setText("SAM point mode pauses during video playback")
        if self.current_frame >= int(self.current_video["frame_count"]) - 1:
            self.seek_frame(0)
        if self.review_segment_start is None:
            self.review_segment_start = self.current_frame
        worker = PlaybackWorker(
            self.current_video,
            self.current_frame,
            float(self.playback_speed.currentData() or 1),
        )
        worker.frame_ready.connect(self.playback_frame_ready)
        worker.failed.connect(lambda message: self.statusBar().showMessage(f"Playback stopped: {message}", 8000))
        worker.finished.connect(self.playback_finished)
        self.playback_worker = worker
        self.play_button.setText("Ⅱ")
        worker.start()
        self._configure_media_controls()

    def playback_frame_ready(self, frame_number: int, image: Any) -> None:
        worker = self.sender()
        if worker is None or worker is not self.playback_worker or not self.current_video:
            return
        self._display_frame(frame_number, image, refresh_maxn=False)

    def playback_finished(self) -> None:
        worker = self.sender()
        if worker is None or worker is not self.playback_worker:
            return
        if worker.last_image is not None and worker.last_frame >= self.current_frame:
            self._display_frame(worker.last_frame, worker.last_image, refresh_maxn=True)
        self.playback_worker = None
        self.play_button.setText("▶")
        self._configure_media_controls()
        if self.sam_checkbox.isChecked():
            self.sam_status.setText(
                f"{self.sam_capability.model_name}: click the fish; "
                "Shift-click or right-click regions that should be excluded"
            )

    def stop_playback(self, *, refresh: bool) -> None:
        worker = self.playback_worker
        if worker is None:
            return
        if worker.isRunning():
            worker.stop()
            worker.wait()
        if worker.last_image is not None and worker.last_frame >= self.current_frame:
            self._display_frame(worker.last_frame, worker.last_image, refresh_maxn=refresh)
        elif refresh and self.current_video:
            self.refresh_frame_annotations()
        self.playback_worker = None
        self.play_button.setText("▶")
        self._configure_media_controls()
        if self.sam_checkbox.isChecked():
            self.sam_status.setText(
                f"{self.sam_capability.model_name}: click the fish; "
                "Shift-click or right-click regions that should be excluded"
            )

    def playback_speed_changed(self) -> None:
        speed = float(self.playback_speed.currentData() or 1)
        if self.playback_worker and self.playback_worker.isRunning():
            self.playback_worker.set_speed(speed)
        self.statusBar().showMessage(f"Playback speed set to {speed:g}×", 3000)

    def _refresh_sam_controls(self) -> None:
        if not hasattr(self, "sam_checkbox"):
            return
        playing = bool(self.playback_worker and self.playback_worker.isRunning())
        seeking = bool(
            self._timeline_scrubbing or self._pending_seek_frame is not None
        )
        active = bool(
            self.sam_checkbox.isChecked()
            and self.sam_capability.available
            and self.current_image is not None
            and not playing
            and not seeking
        )
        has_points = bool(self.sam_points)
        point_mode = active and not self.sam_manual_override
        self.sam_undo_button.setEnabled(point_mode and has_points)
        self.sam_reset_button.setEnabled(point_mode and has_points)
        self.sam_manual_button.setText(
            "Return to SAM points" if self.sam_manual_override else "Use one manual box"
        )
        self.sam_manual_button.setEnabled(active)
        self.sam_accept_button.setEnabled(point_mode and self.sam_result is not None)

    def sam_assist_toggled(self, enabled: bool) -> None:
        if enabled and not self.sam_capability.available:
            with QSignalBlocker(self.sam_checkbox):
                self.sam_checkbox.setChecked(False)
            self.sam_status.setText(self.sam_capability.message)
            self.canvas.set_sam_mode(False)
            self.sam_manual_override = False
            self._refresh_sam_controls()
            return
        if enabled:
            self.stop_playback(refresh=False)
            self.sam_manual_override = False
            self.canvas.set_sam_mode(True)
            self.sam_status.setText(
                f"{self.sam_capability.model_name}: click the fish; "
                "Shift-click or right-click regions that should be excluded"
            )
        else:
            self.sam_manual_override = False
            self.reset_sam_preview()
            self.canvas.set_sam_mode(False)
            self.sam_status.setText(self.sam_capability.message)
        self._refresh_sam_controls()

    def sam_point_added(self, point: tuple[float, float], label: int) -> None:
        if not self.sam_checkbox.isChecked() or self.current_image is None or not self.current_video:
            return
        if not self.selected_species_id():
            QMessageBox.information(
                self,
                "Select a species",
                "Select the fish species on the left before using SAM.",
            )
            return
        self.sam_context = (str(self.current_video["id"]), int(self.current_frame))
        self.sam_points.append((float(point[0]), float(point[1]), int(bool(label))))
        self.sam_result = None
        self.sam_revision += 1
        self.canvas.set_sam_preview(None, self.sam_points)
        self._refresh_sam_controls()
        if not any(item[2] for item in self.sam_points):
            self.sam_status.setText("Add a normal positive click on the fish first")
            return
        self._request_sam_mask()

    def undo_sam_point(self, *_args: Any) -> None:
        if not self.sam_points:
            return
        self.sam_points.pop()
        self.sam_result = None
        self.sam_revision += 1
        self.canvas.set_sam_preview(None, self.sam_points)
        self._refresh_sam_controls()
        if any(item[2] for item in self.sam_points):
            self._request_sam_mask()
        else:
            self.sam_status.setText("Click the fish to create a new mask")

    def reset_sam_preview(self, *_args: Any) -> None:
        self.sam_points = []
        self.sam_result = None
        self.sam_context = None
        self.sam_revision += 1
        self._sam_rerun_requested = False
        if hasattr(self, "canvas"):
            self.canvas.set_sam_preview(None, [])
        if hasattr(self, "sam_status") and self.sam_checkbox.isChecked():
            self.sam_status.setText("Click the fish to create a new mask")
        self._refresh_sam_controls()

    def use_manual_box_mode(self, *_args: Any) -> None:
        if not self.sam_checkbox.isChecked():
            return
        if self.sam_manual_override:
            self.sam_manual_override = False
            self.canvas.set_sam_mode(True)
            self.sam_status.setText("SAM point mode restored. Click a fish.")
        else:
            self.reset_sam_preview()
            self.sam_manual_override = True
            self.canvas.set_sam_mode(False)
            self.sam_status.setText(
                "Draw one manual bounding box. SAM point mode will resume automatically afterward."
            )
        self._refresh_sam_controls()

    def _request_sam_mask(self) -> None:
        if self.sam_worker and self.sam_worker.isRunning():
            self._sam_rerun_requested = True
            self.sam_status.setText("SAM correction queued…")
            return
        if self.current_image is None or not any(item[2] for item in self.sam_points):
            return
        revision = self.sam_revision
        worker = SamMaskWorker(
            self.sam_engine,
            self.current_image,
            [(x, y) for x, y, _label in self.sam_points],
            [label for _x, _y, label in self.sam_points],
            revision,
        )
        worker.completed.connect(self.sam_mask_ready)
        worker.failed.connect(self.sam_mask_failed)
        worker.finished.connect(self.sam_worker_finished)
        self.sam_worker = worker
        self._sam_rerun_requested = False
        self.sam_status.setText(
            f"{self.sam_capability.model_name} is outlining the fish in the background…"
        )
        worker.start()

    def sam_mask_ready(self, revision: int, result: SamMaskResult) -> None:
        if revision != self.sam_revision or not self.sam_checkbox.isChecked():
            return
        self.sam_result = result
        self.canvas.set_sam_preview(result.mask, self.sam_points)
        self.sam_status.setText(
            "Review the cyan mask. Add clicks to include missed areas, "
            "Shift/right-click to exclude errors, then accept."
        )
        self._refresh_sam_controls()

    def sam_mask_failed(self, revision: int, message: str) -> None:
        if revision != self.sam_revision:
            return
        self.sam_result = None
        self.sam_status.setText(f"SAM could not create this mask: {message}")
        self.statusBar().showMessage(
            "SAM failed for this fish; add another point or use a manual box",
            8000,
        )
        self._refresh_sam_controls()

    def sam_worker_finished(self) -> None:
        worker = self.sender()
        if worker is not self.sam_worker:
            return
        self.sam_worker = None
        worker.deleteLater()
        if (
            self._sam_rerun_requested
            and self.sam_checkbox.isChecked()
            and any(item[2] for item in self.sam_points)
        ):
            self._request_sam_mask()

    def accept_sam_mask(self, *_args: Any) -> None:
        if (
            self.sam_result is None
            or self.current_video is None
            or self.current_image is None
            or not self._persist_selected_annotation()
        ):
            return
        species_id = self.selected_species_id()
        if not species_id:
            QMessageBox.information(self, "Select a species", "Select the fish species first.")
            return
        result = self.sam_result
        try:
            annotation = self.db.add_annotation(
                video_id=self.current_video["id"],
                frame_number=self.current_frame,
                time_seconds=self.current_frame / max(0.001, self._fps()),
                species_id=species_id,
                track_id=self.db.next_track_id(self.current_video["id"], species_id),
                box=result.box,
                mask_rle=result.mask_rle,
                status="verified",
                source="manual",
                confidence=result.confidence,
                created_by=self.current_project.get("observer", "") if self.current_project else "",
                life_stage="Adult",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Could not save SAM annotation", str(exc))
            return
        self.reset_sam_preview()
        self.selected_annotation_id = annotation["id"]
        self.refresh_frame_annotations()
        self.sam_status.setText(
            f"Saved {annotation['common_name']} mask and box. Click the next fish."
        )

    def refresh_species(self) -> None:
        query = self.species_search.text().strip().lower() if hasattr(self, "species_search") else ""
        species = [item for item in self.db.list_species() if query in f"{item['common_name']} {item['scientific_name']} {item['code']}".lower()]
        selected = self.species_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.species_list.currentItem() else None
        editor_species = self.annotation_species.currentData() if self.selected_annotation_id else None
        self.species_list.clear()
        for item in species:
            scientific = str(item["scientific_name"] or "").strip()
            common = str(item["common_name"]).strip()
            details = [item["code"]]
            if scientific and scientific.casefold() != common.casefold():
                details.insert(0, scientific)
            row = QListWidgetItem(f"{common}\n{' · '.join(details)}")
            row.setData(Qt.ItemDataRole.UserRole, item["id"])
            row.setForeground(QColor(item["color"]))
            self.species_list.addItem(row)
            if item["id"] == selected:
                self.species_list.setCurrentItem(row)
        if self.species_list.count() and self.species_list.currentRow() < 0:
            self.species_list.setCurrentRow(0)
        self._loading_annotation_editor = True
        try:
            self.annotation_species.clear()
            for item in self.db.list_species():
                self.annotation_species.addItem(f"{item['common_name']} · {item['code']}", item["id"])
            if self.selected_annotation_id is not None and editor_species:
                self.annotation_species.setCurrentIndex(self.annotation_species.findData(editor_species))
        finally:
            self._loading_annotation_editor = False
        if self.selected_annotation_id is None:
            self._show_active_species_in_editor()
        self.refresh_focus_species()

    def refresh_focus_species(self, *_args: Any) -> None:
        if not hasattr(self, "focus_species_list"):
            return
        query = self.focus_species_search.text().strip().casefold()
        selected_species_id = self.selected_species_id()
        species = self.db.list_species()
        matches = [
            item
            for item in species
            if query
            in (
                f"{item['common_name']} {item['scientific_name']} "
                f"{item['code']}"
            ).casefold()
        ]
        with QSignalBlocker(self.focus_species_list):
            self.focus_species_list.clear()
            for item in matches:
                scientific = str(item["scientific_name"] or "").strip()
                common = str(item["common_name"]).strip()
                details = [item["code"]]
                if scientific and scientific.casefold() != common.casefold():
                    details.insert(0, scientific)
                row = QListWidgetItem(f"{common}\n{' - '.join(details)}")
                row.setData(Qt.ItemDataRole.UserRole, item["id"])
                row.setForeground(QColor(item["color"]))
                self.focus_species_list.addItem(row)
                if item["id"] == selected_species_id:
                    self.focus_species_list.setCurrentItem(row)
        selected_species = next(
            (item for item in species if item["id"] == selected_species_id),
            None,
        )
        self.focus_species_current.setText(
            (
                f"Selected: {selected_species['common_name']} - "
                f"{selected_species['code']}"
            )
            if selected_species
            else "Select a species before drawing"
        )

    def focus_species_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        species_id = current.data(Qt.ItemDataRole.UserRole)
        if self.species_search.text():
            with QSignalBlocker(self.species_search):
                self.species_search.clear()
            self.refresh_species()
        self._select_species_in_list(species_id)
        species = self.db.get_species(species_id)
        self.focus_species_current.setText(
            f"Selected: {species['common_name']} - {species['code']}"
        )
        self.statusBar().showMessage(
            f"Next box species: {species['common_name']}",
            3000,
        )

    def active_species_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self.edit_species_button.setEnabled(current is not None)
        if hasattr(self, "focus_species_list"):
            species_id = (
                current.data(Qt.ItemDataRole.UserRole)
                if current is not None
                else None
            )
            with QSignalBlocker(self.focus_species_list):
                for index in range(self.focus_species_list.count()):
                    item = self.focus_species_list.item(index)
                    if item.data(Qt.ItemDataRole.UserRole) == species_id:
                        self.focus_species_list.setCurrentItem(item)
                        break
            if current is not None:
                self.focus_species_current.setText(
                    f"Selected: {current.text().splitlines()[0]}"
                )
        if self.selected_annotation_id is None:
            self._show_active_species_in_editor(current)

    def _show_active_species_in_editor(self, item: QListWidgetItem | None = None) -> None:
        item = item or self.species_list.currentItem()
        species_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        index = self.annotation_species.findData(species_id) if species_id else -1
        self.annotation_autosave_timer.stop()
        self._annotation_editor_dirty = False
        self._loading_annotation_editor = True
        try:
            if index >= 0:
                self.annotation_species.setCurrentIndex(index)
                species_name = item.text().splitlines()[0]
                self.annotation_editor_status.setText(f"Next box species: {species_name}")
            else:
                self.annotation_editor_status.setText("Select a species on the left before drawing")
            self.annotation_track.clear()
            self.annotation_stage.setCurrentText("Adult")
            self.annotation_activity.setCurrentText("Passing")
            self.annotation_uncertain.setChecked(False)
        finally:
            self._loading_annotation_editor = False
        for control in (
            self.annotation_species,
            self.annotation_track,
            self.annotation_stage,
            self.annotation_activity,
            self.annotation_uncertain,
        ):
            control.setEnabled(False)

    def annotation_editor_changed(self, *_args: Any) -> None:
        if self._loading_annotation_editor or not self.selected_annotation_id:
            return
        self._annotation_editor_dirty = True
        self.annotation_editor_status.setText("Editing selected box · saving automatically…")
        self.annotation_autosave_timer.start()

    def autosave_annotation_changes(self, *_args: Any) -> None:
        self._persist_selected_annotation()

    def _persist_selected_annotation(self) -> bool:
        if not self.selected_annotation_id or not self._annotation_editor_dirty:
            return True
        annotation_id = self.selected_annotation_id
        try:
            before = self.db.get_annotation(annotation_id)
            updated = self.db.update_annotation(
                annotation_id,
                species_id=self.annotation_species.currentData(),
                track_id=self.annotation_track.text().strip() or before["track_id"],
                life_stage=self.annotation_stage.currentText(),
                activity=self.annotation_activity.currentText(),
                uncertain=int(self.annotation_uncertain.isChecked()),
            )
        except Exception as exc:
            self.annotation_editor_status.setText("Could not save changes")
            QMessageBox.warning(self, "Could not save annotation", str(exc))
            return False
        self.annotation_autosave_timer.stop()
        self._annotation_editor_dirty = False
        changed = any(
            before[key] != updated[key]
            for key in ("species_id", "track_id", "life_stage", "activity", "uncertain")
        )
        self.annotation_editor_status.setText(
            f"Editing selected box: {updated['common_name']} · changes saved automatically"
        )
        for row in range(self.annotation_table.rowCount()):
            table_item = self.annotation_table.item(row, 0)
            if table_item and table_item.data(Qt.ItemDataRole.UserRole) == annotation_id:
                self.annotation_table.item(row, 1).setText(updated["common_name"])
                self.annotation_table.item(row, 2).setText(updated["track_id"])
                break
        if changed and self.current_video:
            annotations = self.db.annotations_for_frame(self.current_video["id"], self.current_frame)
            self.canvas.set_annotations(annotations)
            verified_count = sum(item["status"] == "verified" for item in annotations)
            pending_count = sum(item["status"] == "pending" for item in annotations)
            frame = self.db.get_frame(self.current_video["id"], self.current_frame)
            with QSignalBlocker(self.frame_complete):
                self.frame_complete.setChecked(bool(frame["reviewed"]))
            completeness = "complete" if frame["reviewed"] else "incomplete"
            self.frame_counts.setText(
                f"Verified fish: {verified_count} · Pending proposals: {pending_count} · {completeness}"
            )
            if not frame["reviewed"]:
                self.training_keyframe_status.setText("Not complete; excluded from final MaxN and training")
            self.refresh_maxn()
        return True

    def add_species(self) -> None:
        dialog = SpeciesDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            if not values["common_name"] or not values["scientific_name"] or not values["code"]:
                raise ValueError("Common name, scientific name and dataset code are required")
            species = self.db.add_species(**values)
            self.species_search.clear()
            self.refresh_species()
            self._select_species_in_list(species["id"])
            self.statusBar().showMessage(f"Added species: {species['common_name']}", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "Could not add species", str(exc))

    def edit_species(self, *_args: Any) -> None:
        species_id = self.selected_species_id()
        if not species_id or not self._persist_selected_annotation():
            return
        species = self.db.get_species(species_id)
        dialog = SpeciesDialog(self, species)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            updated = self.db.update_species(
                species_id,
                common_name=values["common_name"],
                scientific_name=values["scientific_name"],
                color=values["color"],
            )
            self.refresh_species()
            self._select_species_in_list(species_id)
            if self.current_video:
                self.refresh_frame_annotations()
            self.statusBar().showMessage(
                f"Updated species: {updated['common_name']} · dataset code unchanged",
                5000,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Could not edit species", str(exc))

    def _select_species_in_list(self, species_id: str) -> None:
        for index in range(self.species_list.count()):
            item = self.species_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == species_id:
                self.species_list.setCurrentItem(item)
                return

    def selected_species_id(self) -> str | None:
        item = self.species_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def create_manual_box(self, box: tuple[float, float, float, float]) -> None:
        if not self.current_video:
            return
        suggestion = None
        if self.current_image is not None:
            try:
                suggestion = self.inference.classify_box(self.current_image, box)
            except InferenceError:
                suggestion = None
        species_id = suggestion["species_id"] if suggestion else self.selected_species_id()
        if not species_id:
            QMessageBox.information(self, "Select a species", "Select a species before drawing a bounding box. Once a detector is active, it can suggest this automatically.")
            return
        annotation = self.db.add_annotation(
            video_id=self.current_video["id"],
            frame_number=self.current_frame,
            time_seconds=self.current_frame / max(0.001, self._fps()),
            species_id=species_id,
            track_id=self.db.next_track_id(self.current_video["id"], species_id),
            box=box,
            status="pending" if suggestion else "verified",
            source="ai" if suggestion else "manual",
            confidence=suggestion["confidence"] if suggestion else None,
            model_id=suggestion["model_id"] if suggestion else None,
            created_by=self.current_project.get("observer", "") if self.current_project else "",
            life_stage="Adult",
        )
        self.selected_annotation_id = annotation["id"]
        self.refresh_frame_annotations()
        if suggestion:
            self.statusBar().showMessage("AI suggested a species for the drawn box — approve or correct it before it is counted", 8000)
        if self.sam_manual_override:
            self.sam_manual_override = False
            self.canvas.set_sam_mode(self.sam_checkbox.isChecked())
            self.sam_status.setText("Manual box saved. SAM point mode restored for the next fish.")
            self._refresh_sam_controls()

    def canvas_box_changed(self, annotation_id: str, box: tuple[float, float, float, float]) -> None:
        self.db.update_annotation(annotation_id, x=box[0], y=box[1], width=box[2], height=box[3])
        self.refresh_frame_annotations()

    def frame_complete_toggled(self, complete: bool) -> None:
        if not self.current_video:
            return
        if not self._persist_selected_annotation():
            return
        try:
            self.db.set_frame_reviewed(self.current_video["id"], self.current_frame, complete)
        except ValueError as exc:
            with QSignalBlocker(self.frame_complete):
                self.frame_complete.setChecked(False)
            QMessageBox.information(self, "Frame is not complete", str(exc))
        self.refresh_frame_annotations()

    def refresh_frame_annotations(self, *, refresh_maxn: bool = True) -> None:
        if not self.current_video:
            self.canvas.set_annotations([])
            return
        if not self._loading_annotation_editor and not self._persist_selected_annotation():
            return
        annotations = self.db.annotations_for_frame(self.current_video["id"], self.current_frame)
        selected_annotation = next(
            (annotation for annotation in annotations if annotation["id"] == self.selected_annotation_id),
            None,
        )
        if self.selected_annotation_id and selected_annotation is None:
            self.annotation_autosave_timer.stop()
            self._annotation_editor_dirty = False
            self.selected_annotation_id = None
        self.canvas.set_annotations(annotations)
        with QSignalBlocker(self.annotation_table):
            self.annotation_table.setRowCount(len(annotations))
            for row, annotation in enumerate(annotations):
                values = [
                    annotation["status"].title(),
                    annotation["common_name"],
                    annotation["track_id"],
                    (
                        "SAM Manual"
                        if annotation.get("mask_rle")
                        else annotation["source"].replace("_", " ").title()
                    ),
                    f"{annotation['confidence'] * 100:.0f}%" if annotation["confidence"] is not None else "—",
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, annotation["id"])
                    self.annotation_table.setItem(row, column, item)
                if annotation["id"] == self.selected_annotation_id:
                    self.annotation_table.selectRow(row)
        if selected_annotation is not None:
            self._load_annotation_editor(selected_annotation)
        else:
            self._show_active_species_in_editor()
        verified_count = sum(item["status"] == "verified" for item in annotations)
        pending_count = sum(item["status"] == "pending" for item in annotations)
        try:
            frame = self.db.get_frame(self.current_video["id"], self.current_frame)
        except KeyError:
            frame = {"reviewed": 0, "training_selected": 0, "training_reason": ""}
        with QSignalBlocker(self.frame_complete):
            self.frame_complete.setChecked(bool(frame["reviewed"]))
        if frame["reviewed"] and frame["training_selected"]:
            reason = str(frame["training_reason"]).replace("_", " ")
            self.training_keyframe_status.setText(f"Complete MaxN frame · selected for training ({reason})")
        elif frame["reviewed"]:
            self.training_keyframe_status.setText("Complete MaxN frame · training skipped as a near-duplicate")
        else:
            self.training_keyframe_status.setText("Not complete; excluded from final MaxN and training")
        completeness = "complete" if frame["reviewed"] else "incomplete"
        self.frame_counts.setText(f"Verified fish: {verified_count} · Pending proposals: {pending_count} · {completeness}")
        self.approve_frame_button.setEnabled(pending_count > 0)
        if refresh_maxn:
            self.refresh_maxn()

    def annotation_selected(self) -> None:
        rows = self.annotation_table.selectionModel().selectedRows()
        if not rows:
            return
        annotation_id = self.annotation_table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        self.select_annotation_by_id(annotation_id)

    def select_annotation_by_id(self, annotation_id: str | None) -> None:
        previous_id = self.selected_annotation_id
        if annotation_id != previous_id and not self._persist_selected_annotation():
            self.canvas.select_annotation(previous_id)
            return
        self.selected_annotation_id = annotation_id
        self.canvas.select_annotation(annotation_id)
        if not annotation_id:
            with QSignalBlocker(self.annotation_table):
                self.annotation_table.clearSelection()
            self._show_active_species_in_editor()
            self.approve_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            return
        annotation = self.db.get_annotation(annotation_id)
        with QSignalBlocker(self.annotation_table):
            for row in range(self.annotation_table.rowCount()):
                item = self.annotation_table.item(row, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == annotation_id:
                    self.annotation_table.selectRow(row)
                    break
        self._load_annotation_editor(annotation)

    def _load_annotation_editor(self, annotation: dict[str, Any]) -> None:
        self.annotation_autosave_timer.stop()
        self._annotation_editor_dirty = False
        self._loading_annotation_editor = True
        self.annotation_editor_status.setText(
            f"Editing selected box: {annotation['common_name']} · changes save automatically"
        )
        for control in (
            self.annotation_species,
            self.annotation_track,
            self.annotation_stage,
            self.annotation_activity,
            self.annotation_uncertain,
        ):
            control.setEnabled(True)
        try:
            self.annotation_species.setCurrentIndex(self.annotation_species.findData(annotation["species_id"]))
            self.annotation_track.setText(annotation["track_id"])
            self.annotation_stage.setCurrentText(annotation["life_stage"])
            self.annotation_activity.setCurrentText(annotation["activity"])
            self.annotation_uncertain.setChecked(bool(annotation["uncertain"]))
        finally:
            self._loading_annotation_editor = False
        pending = annotation["status"] == "pending"
        self.approve_button.setEnabled(pending)
        self.reject_button.setEnabled(pending)

    def approve_annotation(self) -> None:
        if not self.selected_annotation_id:
            return
        if not self._persist_selected_annotation():
            return
        self.db.review_annotation(self.selected_annotation_id, "approve")
        self.refresh_frame_annotations()

    def reject_annotation(self) -> None:
        if not self.selected_annotation_id:
            return
        self.db.review_annotation(self.selected_annotation_id, "reject")
        self.selected_annotation_id = None
        self.refresh_frame_annotations()

    def approve_frame_proposals(self) -> None:
        if not self.current_video:
            return
        if not self._persist_selected_annotation():
            return
        pending = [item for item in self.db.annotations_for_frame(self.current_video["id"], self.current_frame) if item["status"] == "pending"]
        if not pending:
            return
        if QMessageBox.question(
            self,
            "Approve frame proposals",
            f"Approve all {len(pending)} unchanged proposals on this frame? Review or modify uncertain boxes individually first.",
        ) != QMessageBox.StandardButton.Yes:
            return
        for annotation in pending:
            self.db.review_annotation(annotation["id"], "approve")
        self.refresh_frame_annotations()

    def approve_watched_segment(self) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video":
            return
        if not self._persist_selected_annotation():
            return
        start = self.review_segment_start if self.review_segment_start is not None else self.current_frame
        end = self.current_frame
        first, last = sorted((start, end))
        pending = self.db.pending_annotations_in_range(self.current_video["id"], first, last)
        if QMessageBox.question(
            self,
            "Approve watched segment",
            f"Confirm that you watched frames {first:,}–{last:,}, the boxes followed the fish correctly, "
            f"and every visible fish was boxed.\n\nApprove {len(pending):,} pending boxes and mark the segment complete for MaxN?",
        ) != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for annotation in pending:
                self.db.review_annotation(annotation["id"], "approve")
            frame_numbers = set(self.db.annotated_frame_numbers_in_range(self.current_video["id"], first, last))
            sample_interval = float(self.db.get_setting("training_sample_interval_seconds", 1.0))
            sample_step = max(1, round(self._fps() * sample_interval))
            frame_numbers.update(range(first, last + 1, sample_step))
            frame_numbers.add(last)
            for frame_number in sorted(frame_numbers):
                self.db.set_frame_reviewed(self.current_video["id"], frame_number, True)
            self.review_segment_start = None
            self.refresh_frame_annotations()
            QMessageBox.information(
                self,
                "Segment approved",
                f"Frames {first:,}–{last:,} now contribute completed observations to MaxN. "
                "FinFrame selected manual/corrected and temporally spaced keyframes for training.",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Could not approve segment", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def go_to_next_pending_frame(self) -> None:
        if not self.current_video:
            return
        frame_number = self.db.next_pending_frame(self.current_video["id"], self.current_frame)
        if frame_number is None:
            self.statusBar().showMessage("No pending proposals remain in this media source", 5000)
            return
        self.seek_frame(frame_number)

    def delete_annotation(self) -> None:
        if not self.selected_annotation_id:
            return
        if QMessageBox.question(self, "Delete annotation", "Delete this bounding box?") != QMessageBox.StandardButton.Yes:
            return
        self.db.delete_annotation(self.selected_annotation_id)
        self.selected_annotation_id = None
        self.refresh_frame_annotations()

    def clear_all_video_boxes(self) -> None:
        if not self.current_video:
            return
        is_image = self.current_video.get("media_type") == "image"
        media_noun = "image" if is_image else "video"
        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.information(
                self,
                "Tracking in progress",
                "Wait for whole-video tracking to finish before clearing its boxes.",
            )
            return
        if self.training.status()["running"]:
            QMessageBox.information(
                self,
                "Training in progress",
                "Wait for the current training run to finish before changing the training dataset.",
            )
            return
        stats = self.db.video_annotation_stats(self.current_video["id"])
        if not stats["total"]:
            QMessageBox.information(
                self,
                "No boxes to clear",
                f"This {media_noun} has no bounding boxes.",
            )
            return
        affected_text = (
            "The image will be marked incomplete and removed from MaxN and future training datasets. "
            "Other images and videos are unchanged."
            if is_image
            else
            f"Affected frames will be marked incomplete and removed from MaxN and future training "
            f"datasets. Other images and videos are unchanged."
        )
        frame_scope = "" if is_image else f" across {stats['frames']:,} frames"
        if QMessageBox.question(
            self,
            f"Clear every box from this {media_noun}?",
            f"Permanently delete all {stats['total']:,} bounding boxes from {self.current_video['file_name']}?\n\n"
            f"This includes {stats['verified']:,} verified, {stats['pending']:,} pending and "
            f"{stats['rejected']:,} rejected boxes{frame_scope}. {affected_text}\n\n"
            "Existing trained model files are not changed. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.stop_playback(refresh=False)
        self.selected_annotation_id = None
        self.review_segment_start = None
        self.reset_sam_preview()
        deleted = self.db.clear_video_annotations(self.current_video["id"])
        self.refresh_frame_annotations()
        self.refresh_training_status()
        self.statusBar().showMessage(
            f"Cleared {deleted['total']:,} boxes from this {media_noun}",
            8000,
        )

    def _add_proposals(self, proposals: list[dict[str, Any]], source: str) -> int:
        if not self.current_video:
            return 0
        existing = self.db.annotations_for_frame(self.current_video["id"], self.current_frame)
        added = 0
        for proposal in proposals:
            if any(item["status"] == "verified" and item["species_id"] == proposal["species_id"] and _iou(item, proposal["box"]) >= 0.7 for item in existing):
                continue
            track_id = proposal.get("track_id") or self.db.next_track_id(self.current_video["id"], proposal["species_id"])
            self.db.add_annotation(
                video_id=self.current_video["id"],
                frame_number=self.current_frame,
                time_seconds=self.current_frame / max(0.001, self._fps()),
                species_id=proposal["species_id"],
                track_id=track_id,
                box=proposal["box"],
                status="pending",
                source=source,
                confidence=proposal.get("confidence"),
                model_id=proposal.get("model_id"),
            )
            added += 1
        return added

    def suggest_current_frame(self) -> None:
        if self.current_image is None or not self.current_video:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.db.delete_pending_proposals(self.current_video["id"], source="ai", frame_number=self.current_frame)
            proposals = self.inference.detect_frame(self.current_image)
            added = self._add_proposals(proposals, "ai")
            self.refresh_frame_annotations()
            self.statusBar().showMessage(f"Imported {added} pending AI proposals — approve or correct before use", 8000)
        except InferenceError as exc:
            QMessageBox.information(self, "AI suggestions unavailable", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def start_tracking(self, tracker: str) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video":
            return
        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.information(self, "Tracking in progress", "Wait for the current tracking run to finish.")
            return
        worker = TrackingWorker(self.db, self.inference, self.current_video, tracker, 0.25, 1)
        worker.progress.connect(lambda value, message: (self.tracking_progress.setValue(value), self.statusBar().showMessage(message)))
        worker.failed.connect(self.tracking_failed)
        worker.completed.connect(self.tracking_completed)
        self.tracking_worker = worker
        self.byte_button.setEnabled(False)
        self.bot_button.setEnabled(False)
        worker.start()

    def tracking_failed(self, message: str) -> None:
        self.byte_button.setEnabled(True)
        self.bot_button.setEnabled(True)
        self.tracking_progress.setValue(0)
        QMessageBox.warning(self, "Tracking failed", message)

    def tracking_completed(self, added: int) -> None:
        self.byte_button.setEnabled(True)
        self.bot_button.setEnabled(True)
        self.tracking_progress.setValue(100)
        self.refresh_frame_annotations()
        QMessageBox.information(self, "Tracking complete", f"{added:,} pending tracked boxes were created. They remain excluded until students approve or correct them.")

    def refresh_maxn(self) -> None:
        if not self.current_video:
            live_rows, final_rows = [], []
        else:
            live_rows = self.db.maxn_summary(self.current_video["id"], reviewed_only=False)
            final_rows = self.db.maxn_summary(self.current_video["id"], reviewed_only=True)
        final_by_species = {item["species_id"]: item for item in final_rows}
        self.maxn_table.setRowCount(len(live_rows))
        for row, item in enumerate(live_rows):
            final = final_by_species.get(item["species_id"])
            values = [
                item["common_name"],
                item["code"],
                item["maxn"],
                final["maxn"] if final else "—",
                item["frame_number"],
                self._timecode(item["time_seconds"]),
            ]
            for column, value in enumerate(values):
                self.maxn_table.setItem(row, column, QTableWidgetItem(str(value)))
        if not live_rows:
            self.maxn_status.setText("No verified boxes yet. Draw boxes or approve proposals to begin Live MaxN.")
        elif not final_rows:
            self.maxn_status.setText(
                "Live MaxN is updating. Complete a frame or approve a watched segment to establish Final MaxN."
            )
        else:
            self.maxn_status.setText(
                "Live MaxN includes every verified box; Final MaxN includes completed frames only. Pending proposals count after approval."
            )

    def refresh_training_status(self) -> None:
        status = self.training.status()
        readiness = self.training.readiness()
        readiness_text = (
            "Ready — press Train now to start"
            if readiness["can_train"]
            else (
                f"Need at least {self.training.policy.minimum_verified} selected boxes "
                f"across {self.training.policy.minimum_classes} species"
            )
        )
        self.training_summary.setText(
            f"{readiness['frames']:,} diverse complete keyframes with {readiness['examples']:,} fish boxes across "
            f"{readiness['videos']} media sources and {readiness['classes']} species · "
            f"{readiness['new_changes']} training-dataset changes since training · {readiness_text}"
        )
        self.dataset_stats.setText(
            f"{readiness['verified_total']:,} verified observation boxes · {readiness['reviewed_frames']:,} complete frames · "
            f"{readiness['frames']:,} selected training keyframes · {readiness['pending']:,} pending proposals"
        )
        self.training_progress.setValue(status["progress"])
        self.training_progress.setFormat(status["message"])
        self.train_now_button.setEnabled(readiness["can_train"] and not status["running"])
        active = self.db.active_model()
        self.active_model_label.setText(
            f"Active model: v{active['version']} · mAP50-95 {active['map50_95']:.3f}" if active and active["map50_95"] is not None
            else (
                f"Active model: v{active['version']}"
                if active
                else "Active model: none — training starts only when Train now is pressed"
            )
        )

    def export_training_data(self, fmt: str) -> None:
        default = str(self.data_dir / "exports" / f"finframe_verified_{fmt}.zip")
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()} dataset", default, "ZIP archive (*.zip)")
        if not path:
            return
        include = QMessageBox.question(self, "Include frame images?", "Include extracted JPEG frames? Choose No for labels and manifests only.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export_dataset(self.db, path, fmt=fmt, include_images=include)
            QMessageBox.information(self, "Dataset exported", f"Verified {fmt.upper()} data was written to:\n{path}")
        except DatasetError as exc:
            QMessageBox.warning(self, "Export unavailable", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def backup_project(self) -> None:
        if not self.current_project:
            return
        default = str(self.data_dir / "exports" / f"{self.current_project['name'].replace(' ', '_')}.finframe.json")
        path, _ = QFileDialog.getSaveFileName(self, "Backup project", default, "FinFrame project (*.finframe.json)")
        if path:
            export_project_backup(self.db, self.current_project["id"], path)
            self.statusBar().showMessage(f"Project backup saved to {path}", 8000)

    def export_contribution(self) -> None:
        if not self.current_project:
            return
        safe_name = self.current_project["name"].replace(" ", "_")
        default = str(self.data_dir / "exports" / f"{safe_name}.finframe.zip")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export student contribution",
            default,
            "FinFrame contribution (*.finframe.zip *.zip)",
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            export_contribution_bundle(self.db, self.current_project["id"], path)
            QMessageBox.information(
                self,
                "Contribution ready",
                "One portable file was created with the project labels, annotated frames and selected negative keyframes. "
                "Send this file to the person maintaining the combined training database.\n\n"
                f"{path}",
            )
        except DatasetError as exc:
            QMessageBox.warning(self, "Export unavailable", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def import_project(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import student contributions or projects",
            "",
            "FinFrame files (*.finframe.zip *.zip *.finframe.json *.json)",
        )
        if not paths:
            return
        imported_projects = []
        embedded_frames = 0
        errors = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        for path in paths:
            try:
                if Path(path).suffix.lower() == ".zip":
                    imported = import_contribution_bundle(self.db, path, self.data_dir)
                    project = imported["project"]
                    embedded_frames += int(imported["embedded_frames"])
                else:
                    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
                    project = self.db.import_project_snapshot(snapshot)
                imported_projects.append(project)
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        QApplication.restoreOverrideCursor()
        if imported_projects:
            self.refresh_projects(imported_projects[-1]["id"])
            QMessageBox.information(
                self, "Contributions imported",
                f"Imported {len(imported_projects):,} project{'s' if len(imported_projects) != 1 else ''} and "
                f"stored {embedded_frames:,} portable frames. Complete selected keyframes now contribute to shared training.",
            )
        if errors:
            QMessageBox.warning(self, "Some imports failed", "\n".join(errors))
