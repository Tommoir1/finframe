import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finframe.database import Database
from finframe.training import TrainingCoordinator


class TrainingPolicyTests(unittest.TestCase):
    def test_manual_training_uses_verified_data_and_requires_an_explicit_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Training survey")
            video = db.add_video(project["id"], root / "source.mp4", duration=4, width=100, height=50, fps=25, frame_count=100)
            species = db.list_species()[:2]
            db.set_setting("training_sample_interval_seconds", 0)
            db.set_setting(
                "training_policy",
                {
                    "minimum_verified": 2,
                    "minimum_classes": 2,
                    "retrain_every_verified": 2,
                    "cooldown_minutes": 0,
                },
            )
            coordinator = TrainingCoordinator(db, root)
            db.add_annotation(video_id=video["id"], frame_number=1, time_seconds=.04, species_id=species[0]["id"], track_id="A-1", box=(.1,.1,.2,.2), status="verified", source="manual")
            db.set_frame_reviewed(video["id"], 1, True)
            db.add_annotation(video_id=video["id"], frame_number=2, time_seconds=.08, species_id=species[1]["id"], track_id="B-1", box=(.2,.2,.2,.2), status="pending", source="ai")
            self.assertFalse(coordinator.readiness()["can_train"])
            self.assertFalse(hasattr(coordinator, "maybe_schedule"))
            pending = db.annotations_for_frame(video["id"], 2)[0]
            db.review_annotation(pending["id"], "approve")
            db.set_frame_reviewed(video["id"], 2, True)
            readiness = coordinator.readiness()
            self.assertTrue(readiness["can_train"])
            self.assertEqual(readiness["new_changes"], 2)
            with patch.object(coordinator, "_train") as train:
                self.assertTrue(coordinator.request_training(reason="test button"))
                self.assertIsNotNone(coordinator._thread)
                coordinator._thread.join(timeout=2)
                train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
