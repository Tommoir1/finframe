import unittest

from finframe.seed_tracking import BoundaryIdentityAllocator, SeedTrackingSession


class FakeFrame:
    shape = (100, 200, 3)


class FakeTracker:
    def __init__(self, updates):
        self.updates = list(updates)
        self.initial_box = None

    def init(self, _image, box):
        self.initial_box = box
        return True

    def update(self, _image):
        return self.updates.pop(0)


def annotation(track_id="FISH-001", box=(0.1, 0.2, 0.2, 0.1)):
    return {
        "id": "ann-1",
        "species_id": "species-1",
        "track_id": track_id,
        "x": box[0],
        "y": box[1],
        "width": box[2],
        "height": box[3],
        "life_stage": "Adult",
        "activity": "Passing",
        "uncertain": False,
    }


class SeedTrackingTests(unittest.TestCase):
    def test_student_box_seeds_pending_geometry_on_the_next_frame(self):
        tracker = FakeTracker([(True, (22, 21, 40, 10))])
        session = SeedTrackingSession(lambda: tracker)
        session.seed(annotation(), FakeFrame(), 10)

        predictions, ended = session.update(FakeFrame(), 11)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].track_id, "FISH-001")
        self.assertEqual(predictions[0].frame_number, 11)
        self.assertAlmostEqual(predictions[0].box[0], 0.11)
        self.assertEqual(ended, [])

    def test_edge_disappearance_permanently_ends_seeded_identity(self):
        tracker = FakeTracker([(True, (188, 20, 20, 10)), (False, (0, 0, 0, 0))])
        session = SeedTrackingSession(lambda: tracker)
        session.seed(annotation(), FakeFrame(), 0)

        predictions, _ = session.update(FakeFrame(), 1)
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0].box[0] + predictions[0].box[2], 1.0)

        predictions, ended = session.update(FakeFrame(), 2)
        self.assertEqual(predictions, [])
        self.assertEqual(ended[0].reason, "left_frame")
        self.assertEqual(session.active_count, 0)

    def test_interior_miss_does_not_claim_the_fish_left_the_frame(self):
        tracker = FakeTracker([(False, (0, 0, 0, 0)), (True, (24, 20, 40, 10))])
        session = SeedTrackingSession(lambda: tracker, maximum_interior_misses=3)
        session.seed(annotation(), FakeFrame(), 0)

        predictions, ended = session.update(FakeFrame(), 1)
        self.assertEqual((predictions, ended), ([], []))
        predictions, ended = session.update(FakeFrame(), 2)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(ended, [])

    def test_detector_id_is_replaced_after_a_boundary_exit(self):
        identities = BoundaryIdentityAllocator("BT")
        first = identities.assign(0, [{"raw_track_id": 7, "box": (0.7, 0.2, 0.2, 0.2)}])[0]
        edge = identities.assign(1, [{"raw_track_id": 7, "box": (0.9, 0.2, 0.1, 0.2)}])[0]
        identities.assign(2, [])
        returned = identities.assign(3, [{"raw_track_id": 7, "box": (0.0, 0.2, 0.1, 0.2)}])[0]

        self.assertEqual(first["track_id"], edge["track_id"])
        self.assertNotEqual(first["track_id"], returned["track_id"])

    def test_in_frame_occlusion_can_keep_the_same_detector_identity(self):
        identities = BoundaryIdentityAllocator("BT")
        first = identities.assign(0, [{"raw_track_id": 4, "box": (0.4, 0.2, 0.1, 0.2)}])[0]
        identities.assign(1, [])
        recovered = identities.assign(2, [{"raw_track_id": 4, "box": (0.42, 0.2, 0.1, 0.2)}])[0]
        self.assertEqual(first["track_id"], recovered["track_id"])


if __name__ == "__main__":
    unittest.main()
