import unittest

from finframe.tracking_identity import BoundaryIdentityAllocator


class TrackingIdentityTests(unittest.TestCase):
    def test_detector_id_is_replaced_after_a_boundary_exit(self):
        identities = BoundaryIdentityAllocator("BT")
        first = identities.assign(
            0,
            [{"raw_track_id": 7, "box": (0.7, 0.2, 0.2, 0.2)}],
        )[0]
        edge = identities.assign(
            1,
            [{"raw_track_id": 7, "box": (0.9, 0.2, 0.1, 0.2)}],
        )[0]
        identities.assign(2, [])
        returned = identities.assign(
            3,
            [{"raw_track_id": 7, "box": (0.0, 0.2, 0.1, 0.2)}],
        )[0]

        self.assertEqual(first["track_id"], edge["track_id"])
        self.assertNotEqual(first["track_id"], returned["track_id"])

    def test_in_frame_occlusion_can_keep_the_same_detector_identity(self):
        identities = BoundaryIdentityAllocator("BT")
        first = identities.assign(
            0,
            [{"raw_track_id": 4, "box": (0.4, 0.2, 0.1, 0.2)}],
        )[0]
        identities.assign(1, [])
        recovered = identities.assign(
            2,
            [{"raw_track_id": 4, "box": (0.42, 0.2, 0.1, 0.2)}],
        )[0]

        self.assertEqual(first["track_id"], recovered["track_id"])


if __name__ == "__main__":
    unittest.main()
