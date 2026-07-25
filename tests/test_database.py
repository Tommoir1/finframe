import tempfile
import unittest
from pathlib import Path

import numpy as np

from finframe.database import Database
from finframe.sam_assist import encode_mask_rle


class DatabaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp.name) / "finframe.sqlite3")
        self.project = self.db.create_project("BRUV survey", deployment_id="D-01", observer="Student One")
        self.video = self.db.add_video(
            self.project["id"], Path(self.temp.name) / "survey.mp4", duration=20, width=1280, height=720, fps=25, frame_count=500
        )
        self.species = self.db.list_species()[:2]

    def tearDown(self):
        self.temp.cleanup()

    def add(self, species_index, frame, status="verified", source="manual"):
        return self.db.add_annotation(
            video_id=self.video["id"], frame_number=frame, time_seconds=frame / 25,
            species_id=self.species[species_index]["id"], track_id=f"FISH-{frame}",
            box=(0.1, 0.2, 0.2, 0.1), status=status, source=source, created_by="Student One"
        )

    def test_pending_ai_is_excluded_until_approved(self):
        self.db.set_setting("training_sample_interval_seconds", 0)
        self.add(0, 10)
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        pending = self.add(0, 20, "pending", "ai")
        self.assertEqual(self.db.frame_counts(self.video["id"], 10)[0]["count"], 1)
        self.assertEqual(self.db.training_stats()["examples"], 1)
        approved = self.db.review_annotation(pending["id"], "approve")
        self.db.set_frame_reviewed(self.video["id"], 20, True)
        self.assertEqual(approved["source"], "ai_verified")
        self.assertEqual(self.db.frame_counts(self.video["id"], 20)[0]["count"], 1)
        self.assertEqual(self.db.training_stats()["examples"], 2)

    def test_pending_review_navigation_wraps_to_first_pending_frame(self):
        self.add(0, 20, "pending", "ai")
        self.add(0, 40, "pending", "tracker")
        self.assertEqual(self.db.next_pending_frame(self.video["id"], 20), 40)
        self.assertEqual(self.db.next_pending_frame(self.video["id"], 40), 20)

    def test_annotated_frame_numbers_are_materialized_before_database_closes(self):
        self.add(0, 10)
        self.add(0, 30, "pending", "tracker")
        self.add(0, 50)

        self.assertEqual(self.db.annotated_frame_numbers_in_range(self.video["id"], 0, 40), [10, 30])

    def test_playback_tracker_proposals_are_inserted_once_per_frame_batch(self):
        proposals = [
            {
                "species_id": self.species[0]["id"],
                "track_id": "FISH-001",
                "box": (.1, .1, .2, .2),
                "life_stage": "Adult",
                "activity": "Passing",
            },
            {
                "species_id": self.species[1]["id"],
                "track_id": "FISH-002",
                "box": (.4, .2, .2, .2),
            },
        ]
        added = self.db.add_pending_tracker_annotations(
            self.video["id"], 12, .48, proposals, created_by="Student One"
        )
        repeated = self.db.add_pending_tracker_annotations(
            self.video["id"], 12, .48, proposals, created_by="Student One"
        )

        annotations = self.db.annotations_for_frame(self.video["id"], 12)
        self.assertEqual(added, 2)
        self.assertEqual(repeated, 0)
        self.assertEqual({item["track_id"] for item in annotations}, {"FISH-001", "FISH-002"})
        self.assertTrue(all(item["status"] == "pending" and item["source"] == "tracker" for item in annotations))

    def test_detector_rerun_preserves_pending_box_seed_proposals(self):
        seed = self.add(0, 10, "pending", "tracker")
        model = self.db.register_model("detector.pt", 0.5, 20, "training-run")
        detector = self.db.add_annotation(
            video_id=self.video["id"], frame_number=20, time_seconds=.8,
            species_id=self.species[0]["id"], track_id="BT-00001",
            box=(.2, .2, .2, .1), status="pending", source="tracker", model_id=model["id"]
        )
        deleted = self.db.delete_pending_proposals(self.video["id"], source="tracker", model_only=True)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.db.get_annotation(seed["id"])["status"], "pending")
        with self.assertRaises(KeyError):
            self.db.get_annotation(detector["id"])

    def test_modified_ai_proposal_is_a_corrected_verified_label(self):
        pending = self.add(0, 20, "pending", "ai")
        self.db.update_annotation(pending["id"], species_id=self.species[1]["id"], width=0.25)
        corrected = self.db.review_annotation(pending["id"], "approve")
        self.assertEqual(corrected["source"], "ai_corrected")
        self.assertEqual(corrected["species_id"], self.species[1]["id"])

    def test_maxn_uses_verified_boxes_only(self):
        self.add(0, 10)
        self.add(0, 20)
        self.add(0, 20)
        self.add(0, 30, "pending", "tracker")
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        self.db.set_frame_reviewed(self.video["id"], 20, True)
        summary = self.db.maxn_summary(self.video["id"])
        self.assertEqual(summary[0]["maxn"], 2)
        self.assertEqual(summary[0]["frame_number"], 20)

    def test_verified_edits_and_deletions_advance_dataset_revision(self):
        annotation = self.add(0, 10)
        first = self.db.training_stats()["verified_revision"]
        self.db.update_annotation(annotation["id"], width=0.3)
        second = self.db.training_stats()["verified_revision"]
        self.db.delete_annotation(annotation["id"])
        third = self.db.training_stats()["verified_revision"]
        self.assertGreater(second, first)
        self.assertGreater(third, second)

    def test_manual_box_edit_discards_a_stale_sam_mask(self):
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[4:12, 8:24] = 1
        annotation = self.db.add_annotation(
            video_id=self.video["id"],
            frame_number=10,
            time_seconds=.4,
            species_id=self.species[0]["id"],
            track_id="SAM-001",
            box=(.2, .2, .4, .4),
            mask_rle=encode_mask_rle(mask),
            status="verified",
            source="ai_verified",
        )
        self.assertTrue(annotation["mask_rle"])
        self.assertEqual(annotation["source"], "manual")

        updated = self.db.update_annotation(annotation["id"], width=.45)

        self.assertEqual(updated["mask_rle"], "")

    def test_existing_sam_annotations_are_migrated_to_manual_training_labels(self):
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[4:12, 8:24] = 1
        annotation = self.db.add_annotation(
            video_id=self.video["id"],
            frame_number=10,
            time_seconds=.4,
            species_id=self.species[0]["id"],
            track_id="SAM-001",
            box=(.2, .2, .4, .4),
            mask_rle=encode_mask_rle(mask),
            status="verified",
            source="manual",
        )
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        with self.db.connect() as db:
            db.execute(
                "UPDATE annotations SET source='ai_corrected' WHERE id=?",
                (annotation["id"],),
            )
            db.execute(
                """UPDATE frames SET training_selected=0,training_reason='near_duplicate'
                   WHERE id=?""",
                (annotation["frame_id"],),
            )

        reopened = Database(self.db.path)
        migrated = reopened.get_annotation(annotation["id"])
        frame = reopened.get_frame(self.video["id"], 10)

        self.assertEqual(migrated["source"], "manual")
        self.assertEqual(frame["training_selected"], 1)
        self.assertEqual(frame["training_reason"], "manual_or_corrected")

    def test_clear_video_annotations_is_scoped_and_invalidates_affected_frames(self):
        verified = self.add(0, 10)
        self.add(1, 20, "pending", "tracker")
        rejected = self.add(0, 30, "pending", "ai")
        self.db.review_annotation(rejected["id"], "reject")
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        self.db.set_frame_reviewed(self.video["id"], 40, True)
        other_video = self.db.add_video(
            self.project["id"], Path(self.temp.name) / "other.mp4",
            duration=1, width=640, height=360, fps=25, frame_count=25,
        )
        other = self.db.add_annotation(
            video_id=other_video["id"], frame_number=0, time_seconds=0,
            species_id=self.species[0]["id"], track_id="OTHER-001",
            box=(.1, .1, .2, .2), status="verified", source="manual",
        )

        self.assertEqual(self.db.video_annotation_stats(self.video["id"])["total"], 3)
        result = self.db.clear_video_annotations(self.video["id"])

        self.assertEqual(result["total"], 3)
        self.assertEqual(result["verified"], 1)
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["rejected"], 1)
        for frame_number in (10, 20, 30):
            self.assertEqual(
                self.db.annotations_for_frame(self.video["id"], frame_number, include_rejected=True),
                [],
            )
        self.assertFalse(self.db.get_frame(self.video["id"], 10)["reviewed"])
        self.assertTrue(self.db.get_frame(self.video["id"], 40)["reviewed"])
        self.assertEqual(self.db.get_annotation(other["id"])["video_id"], other_video["id"])
        with self.assertRaises(KeyError):
            self.db.get_annotation(verified["id"])

    def test_project_backup_import_adds_verified_labels_to_shared_database(self):
        self.add(0, 10)
        self.add(1, 20, "pending", "ai")
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        snapshot = self.db.project_snapshot(self.project["id"])
        imported_db = Database(Path(self.temp.name) / "imported.sqlite3")
        imported = imported_db.import_project_snapshot(snapshot)
        self.assertEqual(imported_db.training_stats()["examples"], 1)
        self.assertEqual(imported_db.training_stats()["pending"], 1)
        self.assertEqual(len(imported_db.list_videos(imported["id"])), 1)

    def test_incomplete_frames_do_not_enter_final_maxn(self):
        self.add(0, 10)
        self.add(0, 10)
        self.add(0, 20)
        self.db.set_frame_reviewed(self.video["id"], 20, True)
        self.assertEqual(self.db.maxn_summary(self.video["id"])[0]["maxn"], 1)
        live = self.db.maxn_summary(self.video["id"], reviewed_only=False)
        self.assertEqual(live[0]["maxn"], 2)
        self.assertEqual(live[0]["frame_number"], 10)
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        self.assertEqual(self.db.maxn_summary(self.video["id"])[0]["maxn"], 2)

    def test_complete_tracker_frames_are_temporally_sampled_for_training(self):
        for frame_number in (0, 10, 30):
            self.add(0, frame_number, "verified", "tracker_verified")
            self.db.set_frame_reviewed(self.video["id"], frame_number, True)
        first = self.db.get_frame(self.video["id"], 0)
        near_duplicate = self.db.get_frame(self.video["id"], 10)
        later = self.db.get_frame(self.video["id"], 30)
        self.assertEqual((first["training_selected"], near_duplicate["training_selected"], later["training_selected"]), (1, 0, 1))
        stats = self.db.training_stats()
        self.assertEqual(stats["verified_total"], 3)
        self.assertEqual(stats["examples"], 2)

    def test_editing_a_complete_frame_removes_it_until_reviewed_again(self):
        annotation = self.add(0, 10)
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        completed_revision = self.db.training_stats()["revision"]
        self.assertEqual(self.db.training_stats()["examples"], 1)
        self.db.update_annotation(annotation["id"], width=.25)
        frame = self.db.get_frame(self.video["id"], 10)
        self.assertFalse(frame["reviewed"])
        self.assertEqual(self.db.training_stats()["examples"], 0)
        self.assertEqual(self.db.training_stats()["revision"], completed_revision)
        self.assertEqual(self.db.maxn_summary(self.video["id"]), [])
        self.db.set_frame_reviewed(self.video["id"], 10, True)
        self.assertGreater(self.db.training_stats()["revision"], completed_revision)


if __name__ == "__main__":
    unittest.main()
