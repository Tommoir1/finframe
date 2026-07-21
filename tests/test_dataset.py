import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from finframe.database import Database
from finframe.dataset import (
    DatasetError,
    build_yolo_training_dataset,
    export_contribution_bundle,
    export_dataset,
    import_contribution_bundle,
)


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = Database(self.root / "finframe.sqlite3")
        project = self.db.create_project("Dataset survey", deployment_id="D-02")
        self.video = self.db.add_video(project["id"], self.root / "source.mp4", duration=4, width=100, height=50, fps=25, frame_count=100)
        self.species = self.db.list_species()[0]

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_only_verified_annotations_without_requiring_images(self):
        self.db.add_annotation(
            video_id=self.video["id"], frame_number=10, time_seconds=0.4, species_id=self.species["id"],
            track_id="FISH-001", box=(0.1, 0.2, 0.3, 0.4), status="verified", source="manual"
        )
        pending = self.db.add_annotation(
            video_id=self.video["id"], frame_number=10, time_seconds=0.4, species_id=self.species["id"],
            track_id="FISH-002", box=(0.5, 0.2, 0.2, 0.2), status="pending", source="ai"
        )
        self.db.review_annotation(pending["id"], "reject")
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        destination = export_dataset(self.db, self.root / "dataset.zip", fmt="coco", include_images=False)
        with zipfile.ZipFile(destination) as archive:
            payload = json.loads(archive.read("annotations/instances.json"))
            self.assertEqual(len(payload["annotations"]), 1)
            self.assertEqual(payload["annotations"][0]["bbox"], [10.0, 10.0, 30.0, 20.0])
            self.assertFalse(any(name.startswith("images/") for name in archive.namelist()))
            self.assertIn("metadata/frame_manifest.csv", archive.namelist())
            self.assertIn("metadata/per_frame_counts.csv", archive.namelist())
            self.assertIn("metadata/observations.csv", archive.namelist())

    def test_yolo_export_uses_normalised_centres(self):
        self.db.add_annotation(
            video_id=self.video["id"], frame_number=12, time_seconds=0.48, species_id=self.species["id"],
            track_id="FISH-001", box=(0.1, 0.2, 0.4, 0.2), status="verified", source="manual"
        )
        self.db.set_frame_reviewed(self.video["id"], 12, True)
        destination = export_dataset(self.db, self.root / "dataset.zip", fmt="yolo", include_images=False)
        with zipfile.ZipFile(destination) as archive:
            label_name = next(name for name in archive.namelist() if name.startswith("labels/"))
            self.assertEqual(archive.read(label_name).decode(), "0 0.300000 0.300000 0.400000 0.200000")

    def test_training_builder_aggregates_verified_frames_across_videos(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV training dependencies are not installed")
        project = self.db.get_project(self.video["project_id"])
        for video_number in range(2):
            path = self.root / f"training_{video_number}.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 32))
            if not writer.isOpened():
                self.skipTest("MJPG VideoWriter is unavailable")
            for frame_number in range(3):
                writer.write(np.full((32, 64, 3), 40 + frame_number * 20, dtype=np.uint8))
            writer.release()
            video = self.db.add_video(project["id"], path, duration=.6, width=64, height=32, fps=5, frame_count=3)
            self.db.add_annotation(
                video_id=video["id"], frame_number=1, time_seconds=.2, species_id=self.species["id"],
                track_id=f"FISH-{video_number}", box=(.1,.1,.3,.3), status="verified", source="manual"
            )
            self.db.set_frame_reviewed(video["id"], 1, True)
        result = build_yolo_training_dataset(self.db, self.root / "training_dataset")
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["split"], {"train": 1, "val": 1})
        self.assertTrue(result["yaml"].is_file())

    def test_training_builder_keeps_selected_zero_fish_frames_as_negatives(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV training dependencies are not installed")
        path = self.root / "positive_and_negative.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (64, 32))
        if not writer.isOpened():
            self.skipTest("MJPG VideoWriter is unavailable")
        writer.write(np.full((32, 64, 3), 60, dtype=np.uint8))
        writer.write(np.full((32, 64, 3), 120, dtype=np.uint8))
        writer.release()
        project = self.db.get_project(self.video["project_id"])
        video = self.db.add_video(project["id"], path, duration=2, width=64, height=32, fps=1, frame_count=2)
        self.db.add_annotation(
            video_id=video["id"], frame_number=0, time_seconds=0, species_id=self.species["id"],
            track_id="FISH-001", box=(.1, .1, .3, .3), status="verified", source="manual"
        )
        self.db.set_frame_reviewed(video["id"], 0, True)
        self.db.set_frame_reviewed(video["id"], 1, True)

        result = build_yolo_training_dataset(self.db, self.root / "negative_training_dataset")
        label_files = sorted((result["path"] / "labels").rglob("*.txt"))
        self.assertEqual(result["frames"], 2)
        self.assertEqual(len(label_files), 2)
        self.assertEqual(sum(not label.read_text(encoding="utf-8") for label in label_files), 1)

    def test_portable_contribution_combines_image_labels_without_original_files(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV training dependencies are not installed")
        project = self.db.get_project(self.video["project_id"])
        source_images = []
        for index in range(2):
            path = self.root / f"student_image_{index}.jpg"
            cv2.imwrite(str(path), np.full((40, 80, 3), 50 + index * 50, dtype=np.uint8))
            media = self.db.add_video(
                project["id"], path, duration=0, width=80, height=40,
                fps=1, frame_count=1, media_type="image"
            )
            self.db.add_annotation(
                video_id=media["id"], frame_number=0, time_seconds=0,
                species_id=self.species["id"], track_id=f"IMAGE-{index + 1:03d}",
                box=(.1, .1, .3, .3), status="verified", source="manual"
            )
            self.db.set_frame_reviewed(media["id"], 0, True)
            source_images.append(path)

        bundle = export_contribution_bundle(self.db, project["id"], self.root / "student.finframe.zip")
        with zipfile.ZipFile(bundle) as archive:
            self.assertIn("project.finframe.json", archive.namelist())
            self.assertEqual(len([name for name in archive.namelist() if name.startswith("frames/")]), 2)
        for path in source_images:
            path.unlink()

        combined_root = self.root / "combined"
        combined_db = Database(combined_root / "finframe.sqlite3")
        imported = import_contribution_bundle(combined_db, bundle, combined_root)
        self.assertEqual(imported["embedded_frames"], 2)
        self.assertEqual(combined_db.training_stats()["examples"], 2)
        result = build_yolo_training_dataset(combined_db, combined_root / "training_dataset")
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["split"], {"train": 1, "val": 1})
        with self.assertRaises(DatasetError):
            import_contribution_bundle(combined_db, bundle, combined_root)


if __name__ == "__main__":
    unittest.main()
