import tempfile
import unittest
from pathlib import Path

from finframe.database import Database
from finframe.training import TrainingCoordinator


class TrainingPolicyTests(unittest.TestCase):
    def test_retraining_uses_verified_dataset_changes_not_pending_proposals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db = Database(root / "finframe.sqlite3")
            project = db.create_project("Training survey")
            video = db.add_video(project["id"], root / "source.mp4", duration=4, width=100, height=50, fps=25, frame_count=100)
            species = db.list_species()[:2]
            coordinator = TrainingCoordinator(db, root)
            coordinator.update_policy(minimum_verified=2, minimum_classes=2, retrain_every_verified=2, cooldown_minutes=0)
            db.add_annotation(video_id=video["id"], frame_number=1, time_seconds=.04, species_id=species[0]["id"], track_id="A-1", box=(.1,.1,.2,.2), status="verified", source="manual")
            db.add_annotation(video_id=video["id"], frame_number=2, time_seconds=.08, species_id=species[1]["id"], track_id="B-1", box=(.2,.2,.2,.2), status="pending", source="ai")
            self.assertFalse(coordinator.readiness()["ready"])
            pending = db.annotations_for_frame(video["id"], 2)[0]
            db.review_annotation(pending["id"], "approve")
            readiness = coordinator.readiness()
            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["new_changes"], 2)


if __name__ == "__main__":
    unittest.main()
