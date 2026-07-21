from __future__ import annotations

import json
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
from .dataset import DatasetError, export_dataset, export_project_backup
from .inference import InferenceEngine, InferenceError
from .training import TrainingCoordinator


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
            self.db.delete_pending_proposals(self.video["id"], source="tracker")
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
    def __init__(self, db: Database, data_dir: Path):
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
        self.tracking_worker: TrackingWorker | None = None
        self.setWindowTitle("FinFrame — MaxN video annotation")
        self.resize(1480, 920)
        self._build_ui()
        self._apply_style()
        self.refresh_projects()
        self.training_timer = QTimer(self)
        self.training_timer.timeout.connect(self.refresh_training_status)
        self.training_timer.start(1000)

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
        toolbar.addWidget(QLabel("  Video  "))
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(300)
        self.video_combo.currentIndexChanged.connect(self.video_changed)
        toolbar.addWidget(self.video_combo)
        add_video = QAction("Add video", self)
        add_video.triggered.connect(self.add_video)
        toolbar.addAction(add_video)
        backup = QAction("Backup project", self)
        backup.triggered.connect(self.backup_project)
        toolbar.addAction(backup)
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
        previous = QPushButton("◀ Frame")
        previous.clicked.connect(lambda: self.seek_frame(self.current_frame - 1))
        following = QPushButton("Frame ▶")
        following.clicked.connect(lambda: self.seek_frame(self.current_frame + 1))
        back = QPushButton("−5 s")
        back.clicked.connect(lambda: self.seek_frame(self.current_frame - round(self._fps() * 5)))
        forward = QPushButton("+5 s")
        forward.clicked.connect(lambda: self.seek_frame(self.current_frame + round(self._fps() * 5)))
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.frame_label = QLabel("Frame 0 · 00:00.000")
        for widget in (self.play_button, previous, following, back, forward):
            controls.addWidget(widget)
        controls.addWidget(self.timeline, 1)
        controls.addWidget(self.frame_label)
        video_layout.addLayout(controls)
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
        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(["Status", "Species", "Track", "Source", "Confidence"])
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.annotation_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.annotation_table.itemSelectionChanged.connect(self.annotation_selected)
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        annotation_layout.addWidget(self.frame_counts)
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
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_annotation)
        action_row.addWidget(self.save_annotation_button, 0, 0, 1, 2)
        action_row.addWidget(self.approve_button, 1, 0)
        action_row.addWidget(self.reject_button, 1, 1)
        action_row.addWidget(self.approve_frame_button, 2, 0, 1, 2)
        action_row.addWidget(next_pending_button, 3, 0, 1, 2)
        action_row.addWidget(delete_button, 4, 0, 1, 2)
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
        self.dataset_stats = QLabel("0 verified boxes · 0 videos · 0 pending proposals")
        export_row = QHBoxLayout()
        export_coco = QPushButton("Export all verified data · COCO")
        export_coco.clicked.connect(lambda: self.export_training_data("coco"))
        export_yolo = QPushButton("Export all verified data · YOLO")
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
        self.train_now_button = QPushButton("Train now from all verified data")
        self.train_now_button.clicked.connect(lambda: self.training.request_training(reason="student requested", force=True))
        training_layout.addWidget(self.training_summary, 0, 0, 1, 3)
        training_layout.addWidget(self.active_model_label, 1, 0, 1, 3)
        training_layout.addWidget(QLabel("Retrain after verified dataset changes"), 2, 0)
        training_layout.addWidget(self.training_threshold, 2, 1)
        training_layout.addWidget(self.train_now_button, 2, 2)
        training_layout.addWidget(self.training_progress, 3, 0, 1, 3)
        tabs.addTab(training_tab, "AI training")
        outer.addWidget(tabs)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(lambda: self.seek_frame(self.current_frame + 1))
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
            self.video_combo.addItem("Select a video…", None)
            for video in videos:
                self.video_combo.addItem(video["file_name"], video["id"])
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

    def video_changed(self, index: int) -> None:
        video_id = self.video_combo.itemData(index)
        if not video_id:
            return
        video = self.db.get_video(video_id)
        if not Path(video["path"]).is_file():
            if QMessageBox.question(self, "Video missing", f"The source video is unavailable at:\n{video['path']}\n\nRelink it now?") != QMessageBox.StandardButton.Yes:
                return
            replacement, _ = QFileDialog.getOpenFileName(self, "Relink source video", "", "Video files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)")
            if not replacement:
                return
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
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(video["path"])
        self.current_video = video
        self.timeline.setRange(0, max(0, int(video["frame_count"]) - 1))
        self.seek_frame(0)
        self.statusBar().showMessage(f"Opened {video['file_name']} · {video['width']}×{video['height']} · {video['fps']:.2f} fps")

    def seek_frame(self, frame_number: int) -> None:
        if not self.capture or not self.current_video:
            return
        frame_number = max(0, min(int(self.current_video["frame_count"]) - 1, int(frame_number)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = self.capture.read()
        if not ok:
            self.play_timer.stop()
            self.play_button.setText("▶")
            return
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
        self.refresh_frame_annotations()

    def toggle_playback(self) -> None:
        if not self.current_video:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("▶")
        else:
            self.play_timer.start(max(10, round(1000 / max(1.0, self._fps()))))
            self.play_button.setText("Ⅱ")

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
        if suggestion:
            self.statusBar().showMessage("AI suggested a species for the drawn box — approve or correct it before it is counted", 8000)
        else:
            self.training.maybe_schedule("verified annotation threshold")

    def canvas_box_changed(self, annotation_id: str, box: tuple[float, float, float, float]) -> None:
        self.db.update_annotation(annotation_id, x=box[0], y=box[1], width=box[2], height=box[3])
        self.refresh_frame_annotations()
        annotation = self.db.get_annotation(annotation_id)
        if annotation["status"] == "verified":
            self.training.maybe_schedule("verified box geometry changed")

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
        self.frame_counts.setText(f"Verified fish: {verified_count} · Pending proposals: {pending_count}")
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
        self.db.update_annotation(
            self.selected_annotation_id,
            species_id=self.annotation_species.currentData(),
            track_id=self.annotation_track.text().strip(),
            life_stage=self.annotation_stage.currentText(),
            activity=self.annotation_activity.currentText(),
            uncertain=int(self.annotation_uncertain.isChecked()),
        )
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
        self.db.review_annotation(self.selected_annotation_id, "reject")
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

    def go_to_next_pending_frame(self) -> None:
        if not self.current_video:
            return
        frame_number = self.db.next_pending_frame(self.current_video["id"], self.current_frame)
        if frame_number is None:
            self.statusBar().showMessage("No pending proposals remain in this video", 5000)
            return
        self.seek_frame(frame_number)

    def delete_annotation(self) -> None:
        if not self.selected_annotation_id:
            return
        if QMessageBox.question(self, "Delete annotation", "Delete this bounding box?") != QMessageBox.StandardButton.Yes:
            return
        annotation = self.db.get_annotation(self.selected_annotation_id)
        self.db.delete_annotation(self.selected_annotation_id)
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
        if not self.current_video:
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
            f"{readiness['examples']:,} verified boxes across {readiness['videos']} videos and {readiness['classes']} species · "
            f"{readiness['new_changes']} verified dataset changes since training · {readiness['pending']:,} pending"
        )
        self.dataset_stats.setText(f"{readiness['examples']:,} verified boxes · {readiness['videos']} videos · {readiness['pending']:,} pending proposals")
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

    def import_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import FinFrame project", "", "FinFrame project (*.finframe.json *.json)")
        if not path:
            return
        try:
            snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
            project = self.db.import_project_snapshot(snapshot)
            self.refresh_projects(project["id"])
            self.training.maybe_schedule("verified project imported")
            QMessageBox.information(self, "Project imported", "Verified annotations from the imported project now contribute to the shared dataset and future training.")
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
