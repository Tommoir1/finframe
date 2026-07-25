import unittest

import numpy as np

from finframe.sam_assist import (
    box_from_mask,
    decode_mask_rle,
    encode_mask_rle,
    mask_area_from_rle,
)


class SamAssistTests(unittest.TestCase):
    def test_mask_rle_round_trips_and_produces_a_normalised_box(self):
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:7, 4:15] = 1

        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded)

        np.testing.assert_array_equal(decoded, mask.astype(bool))
        self.assertEqual(mask_area_from_rle(encoded), 55)
        self.assertEqual(box_from_mask(mask), (0.2, 0.2, 0.55, 0.5))

    def test_invalid_mask_rle_is_ignored_safely(self):
        self.assertIsNone(decode_mask_rle(""))
        self.assertIsNone(decode_mask_rle('{"size":[2,2],"counts":[99]}'))
        self.assertEqual(mask_area_from_rle("not json"), 0)


if __name__ == "__main__":
    unittest.main()
