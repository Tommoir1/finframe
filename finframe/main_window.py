from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
from PySide6.QtCore import QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence
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
    QSpinBox,
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
from .seed_tracking import SeedTrackingSession, SeedTrackingUnavailable
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
    """Decode video and propagate seeded boxes without blocking the Qt event loop."""

    frame_ready = Signal(int, object)
    tracking_status = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        db: Database,
        video: dict[str, Any],
        start_frame: int,
        speed: float,
        seed_tracking: SeedTrackingSession,
        created_by: str,
    ):
        super().__init__()
        self.db = db
        self.video = video
        self.start_frame = int(start_frame)
        self.seed_tracking = seed_tracking
        self.created_by = created_by
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

                ended = []
                if self.seed_tracking.active_count:
                    predictions, ended = self.seed_tracking.update(image, frame_number)
                    self.db.add_pending_tracker_annotations(
                        self.video["id"],
                        frame_number,
                        frame_number / fps,
                        ({
                            "species_id": prediction.species_id,
                            "track_id": prediction.track_id,
                            "box": prediction.box,
                            "life_stage": prediction.life_stage,
                            "activity": prediction.activity,
                            "uncertain": prediction.uncertain,
                        } for prediction in predictions),
                        created_by=self.created_by,
                    )
                exited = sum(item.reason == "left_frame" for item in ended)
                if exited:
                    self.tracking_status.emit(
                        f"{exited} track{'s' if exited != 1 else ''} ended at the frame boundary; any return will receive a new identity"
                    )
                elif ended:
                    self.tracking_status.emit(
                        f"{len(ended)} uncertain track{'s' if len(ended) != 1 else ''} stopped"
                    )

                now = perf_counter()
                if now - last_emitted_at >= 1 / 30 or frame_number == frame_count - 1:
                    self.frame_ready.emit(frame_number, image)
                    last_emitted_at = now

                speed = self.speed()
                if self.seed_tracking.active_count or speed < 1:
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
        self.seed_tracking = SeedTrackingSession()
        self.setWindowTitle("FinFrame — MaxN video annotation")
        self.resize(1480, 920)
        self._build_ui()
        self._apply_style()
        self.refresh_projects()
        self.training_timer = QTimer(self)
        self.training_timer.timeout.connect(self.refresh_training_status)
        self.training_timer.start(1000)
        if show_startup_prompt:
            QTimer.singleShot(0, self.choose_startup_task)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Project")
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        species_panel = QGroupBox("Species taxonomy")
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
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.boxCreated.connect(self.create_manual_box)
        self.canvas.boxChanged.connect(self.canvas_box_changed)
        self.canvas.selectionChanged.connect(self.select_annotation_by_id)
        video_layout.addWidget(self.canvas, 1)
        self.timeline_row = QWidget()
        timeline_layout = QHBoxLayout(self.timeline_row)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.frame_label = QLabel("Frame 0 · 00:00.000")
        timeline_layout.addWidget(self.timeline, 1)
        timeline_layout.addWidget(self.frame_label)
        video_layout.addWidget(self.timeline_row)
        controls = QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.clicked.connect(self.toggle_playback)
        self.previous_button = QPushButton("◀ Frame")
        self.previous_button.clicked.connect(lambda: self.navigate_relative(-1))
        self.following_button = QPushButton("Frame ▶")
        self.following_button.clicked.connect(lambda: self.navigate_relative(1))
        self.back_button = QPushButton("−5 s")
        self.back_button.clicked.connect(lambda: self.seek_video_relative(-round(self._fps() * 5)))
        self.forward_button = QPushButton("+5 s")
        self.forward_button.clicked.connect(lambda: self.seek_video_relative(round(self._fps() * 5)))
        for widget in (self.play_button, self.previous_button, self.following_button, self.back_button, self.forward_button):
            controls.addWidget(widget)
        controls.addWidget(QLabel("Speed"))
        self.playback_speed = QComboBox()
        for speed in (0.5, 1, 1.5, 2, 3, 4, 5, 6):
            self.playback_speed.addItem(f"{speed:g}×", speed)
        self.playback_speed.setCurrentIndex(self.playback_speed.findData(1))
        self.playback_speed.currentIndexChanged.connect(self.playback_speed_changed)
        controls.addWidget(self.playback_speed)
        controls.addStretch(1)
        video_layout.addLayout(controls)
        seed_controls = QHBoxLayout()
        self.seed_tracking_checkbox = QCheckBox("Enable box propagation while playing (experimental)")
        self.seed_tracking_checkbox.setToolTip(
            "Off by default. Enable only for continuous playback segments that you will review."
        )
        self.seed_tracking_checkbox.setChecked(False)
        self.seed_tracking_checkbox.toggled.connect(self.seed_tracking_toggled)
        stop_seed_tracking = QPushButton("Stop propagation")
        stop_seed_tracking.clicked.connect(self.stop_seed_tracking)
        self.seed_tracking_status = QLabel("Propagation off")
        seed_controls.addWidget(self.seed_tracking_checkbox)
        seed_controls.addWidget(stop_seed_tracking)
        seed_controls.addWidget(self.seed_tracking_status)
        seed_controls.addStretch(1)
        video_layout.addLayout(seed_controls)
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
        self.clear_video_boxes_button = QPushButton("Clear all boxes from this video")
        self.clear_video_boxes_button.setObjectName("dangerButton")
        self.clear_video_boxes_button.setToolTip("Permanently delete every bounding box from the selected video")
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
        splitter.setSizes([230, 830, 420])

        tabs = QTabWidget()
        tabs.setMaximumHeight(245)
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
        self.active_model_label = QLabel("Active model: none")
        self.training_threshold = QSpinBox()
        self.training_threshold.setRange(1, 500)
        self.training_threshold.setValue(self.training.policy.retrain_every_verified)
        self.training_threshold.valueChanged.connect(self.training_threshold_changed)
        self.training_progress = QProgressBar()
        self.train_now_button = QPushButton("Train now from selected keyframes")
        self.train_now_button.clicked.connect(lambda: self.training.request_training(reason="student requested", force=True))
        training_layout.addWidget(self.training_summary, 0, 0, 1, 3)
        training_layout.addWidget(self.active_model_label, 1, 0, 1, 3)
        training_layout.addWidget(QLabel("Retrain after selected-keyframe changes"), 2, 0)
        training_layout.addWidget(self.training_threshold, 2, 1)
        training_layout.addWidget(self.train_now_button, 2, 2)
        training_layout.addWidget(self.training_progress, 3, 0, 1, 3)
        tabs.addTab(training_tab, "AI training")
        outer.addWidget(tabs)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        delete_shortcut = QAction(self)
        delete_shortcut.setShortcut(QKeySequence.StandardKey.Delete)
        delete_shortcut.triggered.connect(self.delete_annotation)
        self.addAction(delete_shortcut)

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
            QGroupBox { font-weight: 700; border: 1px solid #cbd8d2; border-radius: 8px; margin-top: 10px; padding-top: 12px; background: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #e5eee9; border: 1px solid #b7cbc2; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { background: #d6e7df; }
            QPushButton:disabled { color: #82918b; background: #edf1ef; }
            QPushButton#dangerButton { color: #8f2f25; background: #fff2ef; border-color: #dca69f; }
            QPushButton#dangerButton:hover { background: #ffe4de; }
            QLineEdit, QComboBox, QSpinBox, QTextEdit { background: white; border: 1px solid #b8c8c1; border-radius: 5px; padding: 5px; }
            QTableWidget, QListWidget { background: white; border: 1px solid #cbd8d2; alternate-background-color: #f4f8f6; }
            QScrollArea#annotationScroll { border: 0; background: transparent; }
            QHeaderView::section { background: #e8efeb; padding: 5px; border: 0; border-bottom: 1px solid #bdccc5; }
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
        self.seed_tracking.clear()
        self.review_segment_start = None
        self._refresh_seed_tracking_status()
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

    def _configure_media_controls(self) -> None:
        playable = bool(
            self.current_video
            and self.current_video.get("media_type") == "video"
            and self.capture
            and self.capture.isOpened()
        )
        is_image = bool(self.current_video and self.current_video.get("media_type") == "image")
        playing = bool(self.playback_worker and self.playback_worker.isRunning())
        for control in (
            self.play_button,
            self.back_button,
            self.forward_button,
            self.timeline,
            self.byte_button,
            self.bot_button,
        ):
            control.setEnabled(playable and (control is self.play_button or not playing))
        self.playback_speed.setEnabled(playable)
        self.seed_tracking_checkbox.setEnabled(playable and not playing)
        self.approve_segment_button.setEnabled(playable and not playing)
        self.clear_video_boxes_button.setEnabled(
            bool(self.current_video and self.current_video.get("media_type") == "video") and not playing
        )
        self.previous_button.setText("◀ Image" if is_image else "◀ Frame")
        self.following_button.setText("Image ▶" if is_image else "Frame ▶")
        self.previous_button.setEnabled(playable or (is_image and self._adjacent_image_index(-1) is not None))
        self.following_button.setEnabled(playable or (is_image and self._adjacent_image_index(1) is not None))
        self.canvas.setEnabled(not playing)
        self.annotation_panel.setEnabled(not playing)

    def _adjacent_image_index(self, direction: int) -> int | None:
        index = self.video_combo.currentIndex() + (1 if direction > 0 else -1)
        while 0 <= index < self.video_combo.count():
            media_type = self.video_combo.itemData(index, Qt.ItemDataRole.UserRole + 1)
            if media_type == "image":
                return index
            index += 1 if direction > 0 else -1
        return None

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
            self.seed_tracking.clear()
            self.review_segment_start = None
            self._refresh_seed_tracking_status("Propagation stopped after seeking")
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

    def _display_frame(self, frame_number: int, image: Any, *, refresh_maxn: bool) -> None:
        self.current_frame = frame_number
        self.current_image = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(qimage)
        with QSignalBlocker(self.timeline):
            self.timeline.setValue(frame_number)
        seconds = frame_number / max(0.001, self._fps())
        self.frame_label.setText(f"Frame {frame_number:,} · {self._timecode(seconds)}")
        self.refresh_frame_annotations(refresh_maxn=refresh_maxn)

    def toggle_playback(self) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video" or not self.capture:
            return
        if self.playback_worker and self.playback_worker.isRunning():
            self.stop_playback(refresh=True)
            return
        if not self._persist_selected_annotation():
            return
        if self.current_frame >= int(self.current_video["frame_count"]) - 1:
            self.seek_frame(0)
        if self.review_segment_start is None:
            self.review_segment_start = self.current_frame
        if self.seed_tracking_checkbox.isChecked():
            self._seed_current_frame_annotations()
        worker = PlaybackWorker(
            self.db,
            self.current_video,
            self.current_frame,
            float(self.playback_speed.currentData() or 1),
            self.seed_tracking,
            self.current_project.get("observer", "") if self.current_project else "",
        )
        worker.frame_ready.connect(self.playback_frame_ready)
        worker.tracking_status.connect(lambda message: self._refresh_seed_tracking_status(message))
        worker.failed.connect(lambda message: self.statusBar().showMessage(f"Playback stopped: {message}", 8000))
        worker.finished.connect(self.playback_finished)
        self.playback_worker = worker
        self.play_button.setText("Ⅱ")
        self._configure_media_controls()
        worker.start()

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
        self._refresh_seed_tracking_status()

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
        self._refresh_seed_tracking_status()

    def playback_speed_changed(self) -> None:
        speed = float(self.playback_speed.currentData() or 1)
        if self.playback_worker and self.playback_worker.isRunning():
            self.playback_worker.set_speed(speed)
        self.statusBar().showMessage(f"Playback speed set to {speed:g}×", 3000)

    def seed_tracking_toggled(self, enabled: bool) -> None:
        if not enabled:
            self.seed_tracking.clear()
            self._refresh_seed_tracking_status("Automatic box propagation disabled")
        else:
            self._refresh_seed_tracking_status("New boxes will propagate during playback")

    def stop_seed_tracking(self) -> None:
        self.seed_tracking_checkbox.setChecked(False)
        self.seed_tracking.clear()
        self._refresh_seed_tracking_status("Box propagation stopped")

    def _refresh_seed_tracking_status(self, message: str | None = None) -> None:
        if hasattr(self, "seed_tracking_status"):
            if not self.seed_tracking_checkbox.isChecked():
                self.seed_tracking_status.setText("Propagation off")
            else:
                count = self.seed_tracking.active_count
                self.seed_tracking_status.setText(f"{count} active seeded track{'s' if count != 1 else ''}")
        if message and self.statusBar():
            self.statusBar().showMessage(message, 5000)

    def _seed_annotation(self, annotation: dict[str, Any]) -> None:
        if (
            not self.seed_tracking_checkbox.isChecked()
            or self.current_image is None
            or not self.current_video
            or self.current_video.get("media_type") != "video"
        ):
            return
        if int(annotation["frame_number"]) != self.current_frame:
            return
        try:
            self.seed_tracking.seed(annotation, self.current_image, self.current_frame)
            self._refresh_seed_tracking_status()
        except SeedTrackingUnavailable as exc:
            self.seed_tracking_checkbox.setChecked(False)
            self._refresh_seed_tracking_status(str(exc))

    def _seed_current_frame_annotations(self) -> None:
        if not self.current_video or self.current_image is None:
            return
        for annotation in self.db.annotations_for_frame(self.current_video["id"], self.current_frame):
            self._seed_annotation(annotation)

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

    def active_species_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self.edit_species_button.setEnabled(current is not None)
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
        if before["track_id"] != updated["track_id"]:
            self.seed_tracking.stop(before["track_id"])
        self._seed_annotation(updated)
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
            if before["status"] == "verified":
                self.training.maybe_schedule("verified annotation corrected")
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
        self._seed_annotation(annotation)
        if suggestion:
            self.statusBar().showMessage("AI suggested a species for the drawn box — approve or correct it before it is counted", 8000)
        else:
            self.training.maybe_schedule("verified annotation threshold")

    def canvas_box_changed(self, annotation_id: str, box: tuple[float, float, float, float]) -> None:
        self.db.update_annotation(annotation_id, x=box[0], y=box[1], width=box[2], height=box[3])
        self.refresh_frame_annotations()
        annotation = self.db.get_annotation(annotation_id)
        self._seed_annotation(annotation)
        if annotation["status"] == "verified":
            self.training.maybe_schedule("verified box geometry changed")

    def frame_complete_toggled(self, complete: bool) -> None:
        if not self.current_video:
            return
        if not self._persist_selected_annotation():
            return
        try:
            frame = self.db.set_frame_reviewed(self.current_video["id"], self.current_frame, complete)
            if frame["reviewed"] and frame["training_selected"]:
                self.training.maybe_schedule("complete training keyframe added")
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
                    annotation["source"].replace("_", " ").title(),
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
        self.training.maybe_schedule("AI proposal approved or corrected")

    def reject_annotation(self) -> None:
        if not self.selected_annotation_id:
            return
        annotation = self.db.get_annotation(self.selected_annotation_id)
        self.db.review_annotation(self.selected_annotation_id, "reject")
        self.seed_tracking.stop(annotation["track_id"])
        self._refresh_seed_tracking_status()
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
        self.training.maybe_schedule("frame proposals approved")

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
            self.training.maybe_schedule("watched segment approved")
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
        annotation = self.db.get_annotation(self.selected_annotation_id)
        self.db.delete_annotation(self.selected_annotation_id)
        self.seed_tracking.stop(annotation["track_id"])
        self._refresh_seed_tracking_status()
        self.selected_annotation_id = None
        self.refresh_frame_annotations()
        if annotation["status"] == "verified":
            self.training.maybe_schedule("verified annotation deleted")

    def clear_all_video_boxes(self) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video":
            return
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
            QMessageBox.information(self, "No boxes to clear", "This video has no bounding boxes.")
            return
        if QMessageBox.question(
            self,
            "Clear every box from this video?",
            f"Permanently delete all {stats['total']:,} bounding boxes from {self.current_video['file_name']}?\n\n"
            f"This includes {stats['verified']:,} verified, {stats['pending']:,} pending and "
            f"{stats['rejected']:,} rejected boxes across {stats['frames']:,} frames. Affected frames will be "
            "marked incomplete and removed from MaxN and future training datasets. Other videos are unchanged.\n\n"
            "Existing trained model files are not changed. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.stop_playback(refresh=False)
        self.seed_tracking.clear()
        self.selected_annotation_id = None
        self.review_segment_start = None
        deleted = self.db.clear_video_annotations(self.current_video["id"])
        self._refresh_seed_tracking_status()
        self.refresh_frame_annotations()
        self.refresh_training_status()
        self.statusBar().showMessage(
            f"Cleared {deleted['total']:,} boxes from {self.current_video['file_name']}",
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
        self.stop_seed_tracking()
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

    def training_threshold_changed(self, value: int) -> None:
        self.training.update_policy(retrain_every_verified=value)

    def refresh_training_status(self) -> None:
        status = self.training.status()
        readiness = self.training.readiness()
        if readiness["ready"] and not status["running"]:
            self.training.maybe_schedule("automatic verified dataset threshold")
            status = self.training.status()
        self.training_summary.setText(
            f"{readiness['frames']:,} diverse complete keyframes with {readiness['examples']:,} fish boxes across "
            f"{readiness['videos']} media sources and {readiness['classes']} species · "
            f"{readiness['new_changes']} training-dataset changes since training"
        )
        self.dataset_stats.setText(
            f"{readiness['verified_total']:,} verified observation boxes · {readiness['reviewed_frames']:,} complete frames · "
            f"{readiness['frames']:,} selected training keyframes · {readiness['pending']:,} pending proposals"
        )
        self.training_progress.setValue(status["progress"])
        self.training_progress.setFormat(status["message"])
        self.train_now_button.setEnabled(not status["running"])
        active = self.db.active_model()
        self.active_model_label.setText(
            f"Active model: v{active['version']} · mAP50-95 {active['map50_95']:.3f}" if active and active["map50_95"] is not None
            else (f"Active model: v{active['version']}" if active else "Active model: none — manual labels will seed the first training run")
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
            self.training.maybe_schedule("completed training keyframes imported")
            QMessageBox.information(
                self, "Contributions imported",
                f"Imported {len(imported_projects):,} project{'s' if len(imported_projects) != 1 else ''} and "
                f"stored {embedded_frames:,} portable frames. Complete selected keyframes now contribute to shared training.",
            )
        if errors:
            QMessageBox.warning(self, "Some imports failed", "\n".join(errors))
