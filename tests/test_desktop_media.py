import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from finframe.database import Database
from finframe.main_window import MainWindow, PlaybackWorker
from finframe.seed_tracking import SeedTrackingSession


class StableTracker:
    def init(self, _image, box):
        self.box = box
        return True

    def update(self, _image):
        return True, self.box


class DesktopMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_image_arrows_navigate_between_photos_and_skip_videos(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Photo survey")
            first_path = root / "first.jpg"
            second_path = root / "second.jpg"
            cv2.imwrite(str(first_path), np.full((24, 32, 3), 60, dtype=np.uint8))
            cv2.imwrite(str(second_path), np.full((24, 32, 3), 120, dtype=np.uint8))
            first = db.add_video(
                project["id"], first_path, duration=0, width=32, height=24,
                fps=1, frame_count=1, media_type="image"
            )
            db.ensure_frame(first["id"], 0, 0)
            db.update_frame(first["id"], 0, image_path=first_path)
            db.add_video(
                project["id"], root / "between.mp4", duration=1, width=32, height=24,
                fps=25, frame_count=25, media_type="video"
            )
            second = db.add_video(
                project["id"], second_path, duration=0, width=32, height=24,
                fps=1, frame_count=1, media_type="image"
            )
            db.ensure_frame(second["id"], 0, 0)
            db.update_frame(second["id"], 0, image_path=second_path)

            window = MainWindow(db, root, show_startup_prompt=False)
            window.refresh_projects(project["id"])
            window.video_combo.setCurrentIndex(window.video_combo.findData(first["id"]))
            self.app.processEvents()

            self.assertEqual(window.following_button.text(), "Image ▶")
            self.assertTrue(window.following_button.isEnabled())
            window.following_button.click()
            self.app.processEvents()
            self.assertEqual(window.current_video["id"], second["id"])
            self.assertTrue(window.previous_button.isEnabled())
            window.previous_button.click()
            self.app.processEvents()
            self.assertEqual(window.current_video["id"], first["id"])
            window.close()

    def test_multi_image_import_opens_first_photo_for_forward_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Imported photos")
            first_path = root / "01-first.jpg"
            second_path = root / "02-second.jpg"
            cv2.imwrite(str(first_path), np.full((24, 32, 3), 60, dtype=np.uint8))
            cv2.imwrite(str(second_path), np.full((24, 32, 3), 120, dtype=np.uint8))
            window = MainWindow(db, root, show_startup_prompt=False)
            window.refresh_projects(project["id"])

            window._import_images([first_path, second_path])
            self.app.processEvents()

            self.assertEqual(Path(window.current_video["path"]), first_path.resolve())
            self.assertTrue(window.following_button.isEnabled())
            window.following_button.click()
            self.app.processEvents()
            self.assertEqual(Path(window.current_video["path"]), second_path.resolve())
            window.close()

    def test_timeline_sits_directly_below_media_and_master_species_has_no_extra_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Layout survey")
            window = MainWindow(db, root, show_startup_prompt=False)
            window.refresh_projects(project["id"])

            video_layout = window.canvas.parentWidget().layout()
            self.assertEqual(video_layout.indexOf(window.timeline_row), video_layout.indexOf(window.canvas) + 1)
            self.assertIs(window.timeline.parentWidget(), window.timeline_row)
            species_texts = [window.species_list.item(index).text() for index in range(window.species_list.count())]
            self.assertTrue(species_texts)
            self.assertFalse(any("Master species list" in text for text in species_texts))
            window.close()

    def test_background_playback_keeps_qt_events_responsive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "playback.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (48, 32))
            if not writer.isOpened():
                self.skipTest("MJPG VideoWriter is unavailable")
            for frame_number in range(60):
                writer.write(np.full((32, 48, 3), frame_number, dtype=np.uint8))
            writer.release()
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Playback survey")
            video = db.add_video(
                project["id"], path, duration=2, width=48, height=32,
                fps=30, frame_count=60, media_type="video"
            )
            worker = PlaybackWorker(db, video, 0, 6, SeedTrackingSession(), "Student")
            event_processed = []
            emitted_frames = []
            worker.frame_ready.connect(lambda frame_number, _image: emitted_frames.append(frame_number))
            QTimer.singleShot(20, lambda: (event_processed.append(True), worker.set_speed(4)))
            worker.start()
            deadline = time.monotonic() + 5
            while worker.isRunning() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.001)
            worker.wait()
            self.app.processEvents()

            self.assertTrue(event_processed)
            self.assertTrue(emitted_frames)
            self.assertEqual(worker.last_frame, 59)

    def test_background_playback_preserves_sequential_seed_tracking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tracked-playback.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (48, 32))
            if not writer.isOpened():
                self.skipTest("MJPG VideoWriter is unavailable")
            first_image = np.full((32, 48, 3), 80, dtype=np.uint8)
            for _ in range(12):
                writer.write(first_image)
            writer.release()
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Tracked playback")
            video = db.add_video(
                project["id"], path, duration=.4, width=48, height=32,
                fps=30, frame_count=12, media_type="video"
            )
            species = db.list_species()[0]
            annotation = db.add_annotation(
                video_id=video["id"], frame_number=0, time_seconds=0,
                species_id=species["id"], track_id="FISH-001", box=(.1, .1, .2, .2),
                status="verified", source="manual", created_by="Student"
            )
            session = SeedTrackingSession(tracker_factory=StableTracker)
            session.seed(annotation, first_image, 0)
            worker = PlaybackWorker(db, video, 0, 6, session, "Student")
            worker.start()
            self.assertTrue(worker.wait(5000))

            self.assertEqual(worker.last_frame, 11)
            for frame_number in range(1, 12):
                proposals = db.annotations_for_frame(video["id"], frame_number)
                self.assertEqual(len(proposals), 1)
                self.assertEqual(proposals[0]["track_id"], "FISH-001")
                self.assertEqual(proposals[0]["status"], "pending")

    def test_window_plays_after_a_student_draws_an_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "annotated-playback.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (48, 32))
            if not writer.isOpened():
                self.skipTest("MJPG VideoWriter is unavailable")
            for _ in range(12):
                writer.write(np.full((32, 48, 3), 80, dtype=np.uint8))
            writer.release()
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Student annotation playback")
            video = db.add_video(
                project["id"], path, duration=.4, width=48, height=32,
                fps=30, frame_count=12, media_type="video"
            )
            window = MainWindow(db, root, show_startup_prompt=False)
            window.seed_tracking = SeedTrackingSession(tracker_factory=StableTracker)
            window.refresh_projects(project["id"])
            window.video_combo.setCurrentIndex(window.video_combo.findData(video["id"]))
            self.app.processEvents()

            window.create_manual_box((.1, .1, .2, .2))
            loaded = db.annotations_for_frame(video["id"], 0)
            self.assertEqual(loaded[0]["frame_number"], 0)
            window.toggle_playback()
            worker = window.playback_worker
            self.assertIsNotNone(worker)
            self.assertTrue(worker.wait(5000))
            self.app.processEvents()

            self.assertEqual(worker.error_message, "")
            self.assertEqual(worker.last_frame, 11)
            self.assertEqual(len(db.annotations_for_frame(video["id"], 1)), 1)
            window.close()

    def test_main_window_can_change_speed_and_pause_during_playback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "window-playback.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (48, 32))
            if not writer.isOpened():
                self.skipTest("MJPG VideoWriter is unavailable")
            for frame_number in range(180):
                writer.write(np.full((32, 48, 3), frame_number % 255, dtype=np.uint8))
            writer.release()
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Window playback")
            video = db.add_video(
                project["id"], path, duration=6, width=48, height=32,
                fps=30, frame_count=180, media_type="video"
            )
            window = MainWindow(db, root, show_startup_prompt=False)
            window.refresh_projects(project["id"])
            window.video_combo.setCurrentIndex(window.video_combo.findData(video["id"]))
            loop = QEventLoop()
            QTimer.singleShot(10, window.play_button.click)
            QTimer.singleShot(
                80,
                lambda: window.playback_speed.setCurrentIndex(window.playback_speed.findData(6.0)),
            )
            QTimer.singleShot(350, window.play_button.click)
            QTimer.singleShot(450, loop.quit)
            loop.exec()
            self.app.processEvents()

            self.assertEqual(window.playback_speed.currentData(), 6.0)
            self.assertGreater(window.current_frame, 0)
            self.assertIsNone(window.playback_worker)
            self.assertTrue(window.project_combo.isEnabled())
            window.close()


if __name__ == "__main__":
    unittest.main()
