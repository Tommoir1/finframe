from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
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
        self.review_segment_start: int | None = None
        self.tracking_worker: TrackingWorker | None = None
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
        toolbar.addWidget(QLabel("  Project  "))
        self.project_combo = QComboBox()
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
        toolbar.addWidget(QLabel("  Media  "))
        self.video_combo = QComboBox()
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
        add_species = QPushButton("Add species")
        add_species.clicked.connect(self.add_species)
        species_layout.addWidget(self.species_search)
        species_layout.addWidget(self.species_list, 1)
        species_layout.addWidget(add_species)
        splitter.addWidget(species_panel)

        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.boxCreated.connect(self.create_manual_box)
        self.canvas.boxChanged.connect(self.canvas_box_changed)
        self.canvas.selectionChanged.connect(self.select_annotation_by_id)
        video_layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.clicked.connect(self.toggle_playback)
        self.previous_button = QPushButton("◀ Frame")
        self.previous_button.clicked.connect(lambda: self.seek_frame(self.current_frame - 1))
        self.following_button = QPushButton("Frame ▶")
        self.following_button.clicked.connect(lambda: self.seek_frame(self.current_frame + 1))
        self.back_button = QPushButton("−5 s")
        self.back_button.clicked.connect(lambda: self.seek_frame(self.current_frame - round(self._fps() * 5)))
        self.forward_button = QPushButton("+5 s")
        self.forward_button.clicked.connect(lambda: self.seek_frame(self.current_frame + round(self._fps() * 5)))
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.frame_label = QLabel("Frame 0 · 00:00.000")
        for widget in (self.play_button, self.previous_button, self.following_button, self.back_button, self.forward_button):
            controls.addWidget(widget)
        controls.addWidget(QLabel("Speed"))
        self.playback_speed = QComboBox()
        for speed in (0.5, 1, 1.5, 2, 3, 4, 5, 6):
            self.playback_speed.addItem(f"{speed:g}×", speed)
        self.playback_speed.setCurrentIndex(self.playback_speed.findData(1))
        self.playback_speed.currentIndexChanged.connect(self.playback_speed_changed)
        controls.addWidget(self.playback_speed)
        controls.addWidget(self.timeline, 1)
        controls.addWidget(self.frame_label)
        video_layout.addLayout(controls)
        seed_controls = QHBoxLayout()
        self.seed_tracking_checkbox = QCheckBox("Propagate drawn boxes while playing")
        self.seed_tracking_checkbox.setChecked(True)
        self.seed_tracking_checkbox.toggled.connect(self.seed_tracking_toggled)
        stop_seed_tracking = QPushButton("Stop propagation")
        stop_seed_tracking.clicked.connect(self.stop_seed_tracking)
        self.seed_tracking_status = QLabel("0 active seeded tracks")
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

        annotation_panel = QGroupBox("Frame annotations")
        annotation_layout = QVBoxLayout(annotation_panel)
        self.frame_counts = QLabel("Verified fish: 0")
        self.frame_complete = QCheckBox("Frame complete — all visible fish are boxed")
        self.frame_complete.toggled.connect(self.frame_complete_toggled)
        self.training_keyframe_status = QLabel("Not complete; excluded from final MaxN and training")
        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(["Status", "Species", "Track", "Source", "Confidence"])
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.annotation_table.itemSelectionChanged.connect(self.annotation_selected)
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        annotation_layout.addWidget(self.frame_counts)
        annotation_layout.addWidget(self.frame_complete)
        annotation_layout.addWidget(self.training_keyframe_status)
        annotation_layout.addWidget(self.annotation_table, 1)
        form = QFormLayout()
        self.annotation_species = QComboBox()
        self.annotation_track = QLineEdit()
        self.annotation_stage = QComboBox()
        self.annotation_stage.addItems(["Adult", "Juvenile", "Unknown"])
        self.annotation_activity = QComboBox()
        self.annotation_activity.addItems(["Passing", "Feeding", "Schooling", "Resting", "Unknown"])
        self.annotation_uncertain = QCheckBox("Uncertain")
        form.addRow("Species", self.annotation_species)
        form.addRow("Track ID", self.annotation_track)
        form.addRow("Life stage", self.annotation_stage)
        form.addRow("Activity", self.annotation_activity)
        form.addRow("", self.annotation_uncertain)
        annotation_layout.addLayout(form)
        action_row = QGridLayout()
        self.save_annotation_button = QPushButton("Save changes")
        self.save_annotation_button.clicked.connect(self.save_annotation_changes)
        self.approve_button = QPushButton("Approve proposal")
        self.approve_button.clicked.connect(self.approve_annotation)
        self.reject_button = QPushButton("Reject proposal")
        self.reject_button.clicked.connect(self.reject_annotation)
        self.approve_frame_button = QPushButton("Approve all unchanged proposals on frame")
        self.approve_frame_button.clicked.connect(self.approve_frame_proposals)
        next_pending_button = QPushButton("Next pending frame")
        next_pending_button.clicked.connect(self.go_to_next_pending_frame)
        self.approve_segment_button = QPushButton("Approve watched segment")
        self.approve_segment_button.clicked.connect(self.approve_watched_segment)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_annotation)
        action_row.addWidget(self.save_annotation_button, 0, 0, 1, 2)
        action_row.addWidget(self.approve_button, 1, 0)
        action_row.addWidget(self.reject_button, 1, 1)
        action_row.addWidget(self.approve_frame_button, 2, 0, 1, 2)
        action_row.addWidget(next_pending_button, 3, 0, 1, 2)
        action_row.addWidget(self.approve_segment_button, 4, 0, 1, 2)
        action_row.addWidget(delete_button, 5, 0, 1, 2)
        annotation_layout.addLayout(action_row)
        splitter.addWidget(annotation_panel)
        splitter.setSizes([240, 900, 330])

        tabs = QTabWidget()
        tabs.setMaximumHeight(245)
        self.maxn_table = QTableWidget(0, 5)
        self.maxn_table.setHorizontalHeaderLabels(["Species", "Code", "MaxN", "Peak frame", "Peak time"])
        self.maxn_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.maxn_table, "MaxN summary")
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

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_playback)
        delete_shortcut = QAction(self)
        delete_shortcut.setShortcut(QKeySequence.StandardKey.Delete)
        delete_shortcut.triggered.connect(self.delete_annotation)
        self.addAction(delete_shortcut)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f3; color: #14251f; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }
            QToolBar { background: #102a22; color: white; spacing: 8px; padding: 6px; border: 0; }
            QToolBar QLabel { color: #d7e5df; }
            QGroupBox { font-weight: 700; border: 1px solid #cbd8d2; border-radius: 8px; margin-top: 10px; padding-top: 12px; background: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #e5eee9; border: 1px solid #b7cbc2; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { background: #d6e7df; }
            QPushButton:disabled { color: #82918b; background: #edf1ef; }
            QLineEdit, QComboBox, QSpinBox, QTextEdit { background: white; border: 1px solid #b8c8c1; border-radius: 5px; padding: 5px; }
            QTableWidget, QListWidget { background: white; border: 1px solid #cbd8d2; alternate-background-color: #f4f8f6; }
            QHeaderView::section { background: #e8efeb; padding: 5px; border: 0; border-bottom: 1px solid #bdccc5; }
            QTabWidget::pane { border: 1px solid #cbd8d2; background: white; }
            QTabBar::tab { padding: 7px 16px; background: #e4ece8; }
            QTabBar::tab:selected { background: white; font-weight: 700; }
            QProgressBar { border: 1px solid #b7cbc2; border-radius: 5px; text-align: center; background: white; }
            QProgressBar::chunk { background: #41a88a; border-radius: 4px; }
        """)

    def closeEvent(self, event: Any) -> None:
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
        last_id = None
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
            last_id = media["id"]
        if last_id:
            self.refresh_videos(last_id)
            self.statusBar().showMessage(
                f"Added {len(paths) - len(unreadable):,} image{'s' if len(paths) - len(unreadable) != 1 else ''}",
                5000,
            )
        if unreadable:
            QMessageBox.warning(self, "Some images were skipped", "OpenCV could not read:\n" + "\n".join(unreadable))

    def video_changed(self, index: int) -> None:
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
        for control in (
            self.play_button,
            self.previous_button,
            self.following_button,
            self.back_button,
            self.forward_button,
            self.playback_speed,
            self.timeline,
            self.seed_tracking_checkbox,
            self.byte_button,
            self.bot_button,
        ):
            control.setEnabled(playable)
        self.approve_segment_button.setEnabled(playable)

    def seek_frame(self, frame_number: int, propagate_seeded: bool = False) -> None:
        if not self.current_video:
            return
        frame_number = max(0, min(int(self.current_video["frame_count"]) - 1, int(frame_number)))
        previous_frame = self.current_frame
        if not propagate_seeded and self.current_image is not None and frame_number != previous_frame:
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
            self.play_timer.stop()
            self.play_button.setText("▶")
            self.statusBar().showMessage("This frame is unavailable; relink the full source video to browse unannotated frames", 8000)
            return
        self.current_frame = frame_number
        self.current_image = image
        if propagate_seeded and frame_number == previous_frame + 1:
            self._propagate_seeded_boxes(image, frame_number)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.canvas.set_frame(qimage)
        with QSignalBlocker(self.timeline):
            self.timeline.setValue(frame_number)
        seconds = frame_number / max(0.001, self._fps())
        self.frame_label.setText(f"Frame {frame_number:,} · {self._timecode(seconds)}")
        self.refresh_frame_annotations()

    def advance_playback(self) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video":
            return
        if self.current_frame >= int(self.current_video["frame_count"]) - 1:
            self.play_timer.stop()
            self.play_button.setText("▶")
            return
        self.seek_frame(
            self.current_frame + 1,
            propagate_seeded=self.seed_tracking_checkbox.isChecked(),
        )

    def toggle_playback(self) -> None:
        if not self.current_video or self.current_video.get("media_type") != "video" or not self.capture:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("▶")
        else:
            if self.review_segment_start is None:
                self.review_segment_start = self.current_frame
            if self.seed_tracking_checkbox.isChecked():
                self._seed_current_frame_annotations()
            self.play_timer.start(self._playback_interval())
            self.play_button.setText("Ⅱ")

    def _playback_interval(self) -> int:
        speed = float(self.playback_speed.currentData() or 1)
        return max(1, round(1000 / max(1.0, self._fps() * speed)))

    def playback_speed_changed(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.start(self._playback_interval())
        speed = float(self.playback_speed.currentData() or 1)
        self.statusBar().showMessage(f"Playback speed set to {speed:g}×", 3000)

    def seed_tracking_toggled(self, enabled: bool) -> None:
        if not enabled:
            self.seed_tracking.clear()
            self._refresh_seed_tracking_status("Automatic box propagation disabled")
        else:
            self._refresh_seed_tracking_status("New boxes will propagate during playback")

    def stop_seed_tracking(self) -> None:
        self.seed_tracking.clear()
        self._refresh_seed_tracking_status("All seeded tracks stopped")

    def _refresh_seed_tracking_status(self, message: str | None = None) -> None:
        if hasattr(self, "seed_tracking_status"):
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

    def _propagate_seeded_boxes(self, image: Any, frame_number: int) -> None:
        if not self.current_video or not self.seed_tracking.active_count:
            return
        predictions, ended = self.seed_tracking.update(image, frame_number)
        existing = self.db.annotations_for_frame(self.current_video["id"], frame_number)
        existing_tracks = {item["track_id"] for item in existing}
        for prediction in predictions:
            if prediction.track_id in existing_tracks:
                continue
            self.db.add_annotation(
                video_id=self.current_video["id"],
                frame_number=frame_number,
                time_seconds=frame_number / max(0.001, self._fps()),
                species_id=prediction.species_id,
                track_id=prediction.track_id,
                box=prediction.box,
                status="pending",
                source="tracker",
                confidence=None,
                model_id=None,
                created_by=self.current_project.get("observer", "") if self.current_project else "",
                life_stage=prediction.life_stage,
                activity=prediction.activity,
                uncertain=prediction.uncertain,
            )
        exited = sum(item.reason == "left_frame" for item in ended)
        message = None
        if exited:
            message = f"{exited} track{'s' if exited != 1 else ''} ended at the frame boundary; any return will receive a new identity"
        elif ended:
            message = f"{len(ended)} uncertain track{'s' if len(ended) != 1 else ''} stopped"
        self._refresh_seed_tracking_status(message)

    def refresh_species(self) -> None:
        query = self.species_search.text().strip().lower() if hasattr(self, "species_search") else ""
        species = [item for item in self.db.list_species() if query in f"{item['common_name']} {item['scientific_name']} {item['code']}".lower()]
        selected = self.species_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.species_list.currentItem() else None
        self.species_list.clear()
        for item in species:
            row = QListWidgetItem(f"{item['common_name']}\n{item['scientific_name'] or 'Unspecified'} · {item['code']}")
            row.setData(Qt.ItemDataRole.UserRole, item["id"])
            row.setForeground(QColor(item["color"]))
            self.species_list.addItem(row)
            if item["id"] == selected:
                self.species_list.setCurrentItem(row)
        if self.species_list.count() and self.species_list.currentRow() < 0:
            self.species_list.setCurrentRow(0)
        self.annotation_species.clear()
        for item in self.db.list_species():
            self.annotation_species.addItem(f"{item['common_name']} · {item['code']}", item["id"])

    def add_species(self) -> None:
        common, ok = QInputDialog.getText(self, "Add species", "Common name")
        if not ok or not common.strip():
            return
        scientific, ok = QInputDialog.getText(self, "Add species", "Scientific name")
        if not ok:
            return
        code, ok = QInputDialog.getText(self, "Add species", "Stable species code")
        if not ok or not code.strip():
            return
        color = QColorDialog.getColor(QColor("#ff8465"), self, "Bounding-box colour")
        if not color.isValid():
            return
        try:
            self.db.add_species(common, scientific, code, color.name())
            self.refresh_species()
        except Exception as exc:
            QMessageBox.warning(self, "Could not add species", str(exc))

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
        try:
            frame = self.db.set_frame_reviewed(self.current_video["id"], self.current_frame, complete)
            if frame["reviewed"] and frame["training_selected"]:
                self.training.maybe_schedule("complete training keyframe added")
        except ValueError as exc:
            with QSignalBlocker(self.frame_complete):
                self.frame_complete.setChecked(False)
            QMessageBox.information(self, "Frame is not complete", str(exc))
        self.refresh_frame_annotations()

    def refresh_frame_annotations(self) -> None:
        if not self.current_video:
            self.canvas.set_annotations([])
            return
        annotations = self.db.annotations_for_frame(self.current_video["id"], self.current_frame)
        self.canvas.set_annotations(annotations)
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
        self.refresh_maxn()

    def annotation_selected(self) -> None:
        rows = self.annotation_table.selectionModel().selectedRows()
        if not rows:
            return
        annotation_id = self.annotation_table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        self.select_annotation_by_id(annotation_id)

    def select_annotation_by_id(self, annotation_id: str | None) -> None:
        self.selected_annotation_id = annotation_id
        self.canvas.select_annotation(annotation_id)
        if not annotation_id:
            self.approve_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            return
        annotation = self.db.get_annotation(annotation_id)
        self.annotation_species.setCurrentIndex(self.annotation_species.findData(annotation["species_id"]))
        self.annotation_track.setText(annotation["track_id"])
        self.annotation_stage.setCurrentText(annotation["life_stage"])
        self.annotation_activity.setCurrentText(annotation["activity"])
        self.annotation_uncertain.setChecked(bool(annotation["uncertain"]))
        pending = annotation["status"] == "pending"
        self.approve_button.setEnabled(pending)
        self.reject_button.setEnabled(pending)

    def save_annotation_changes(self) -> None:
        if not self.selected_annotation_id:
            return
        before = self.db.get_annotation(self.selected_annotation_id)
        updated = self.db.update_annotation(
            self.selected_annotation_id,
            species_id=self.annotation_species.currentData(),
            track_id=self.annotation_track.text().strip(),
            life_stage=self.annotation_stage.currentText(),
            activity=self.annotation_activity.currentText(),
            uncertain=int(self.annotation_uncertain.isChecked()),
        )
        if before["track_id"] != updated["track_id"]:
            self.seed_tracking.stop(before["track_id"])
        self._seed_annotation(updated)
        self.refresh_frame_annotations()
        if before["status"] == "verified":
            self.training.maybe_schedule("verified annotation corrected")

    def approve_annotation(self) -> None:
        if not self.selected_annotation_id:
            return
        self.save_annotation_changes()
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
        rows = self.db.maxn_summary(self.current_video["id"]) if self.current_video else []
        self.maxn_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [item["common_name"], item["code"], item["maxn"], item["frame_number"], self._timecode(item["time_seconds"])]
            for column, value in enumerate(values):
                self.maxn_table.setItem(row, column, QTableWidgetItem(str(value)))

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
