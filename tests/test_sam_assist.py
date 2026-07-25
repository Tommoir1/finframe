import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from finframe.sam_assist import (
    SamAssistEngine,
    box_from_mask,
    decode_mask_rle,
    encode_mask_rle,
    mask_area_from_rle,
)


class TensorStub:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __len__(self):
        return len(self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class PromptAwareSamStub:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        mask = np.zeros(kwargs["source"].shape[:2], dtype=np.float32)
        mask[2:8, 2:8] = 1
        if len(kwargs["points"][0]) > 1:
            if kwargs["labels"][0][-1] == 0:
                mask[2:8, 5:8] = 0
            else:
                mask[0:2, 2:8] = 1
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=TensorStub(mask[None, ...])),
                boxes=None,
            )
        ]


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

    def test_positive_and_negative_points_are_grouped_to_refine_one_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = SamAssistEngine(Path(temporary))
            model = PromptAwareSamStub()
            engine._model = model
            image = np.zeros((10, 20, 3), dtype=np.uint8)

            initial = engine.segment(image, [(.2, .4)], [1])
            expanded = engine.segment(image, [(.2, .4), (.4, .4)], [1, 1])
            corrected = engine.segment(image, [(.2, .4), (.4, .4)], [1, 0])

            self.assertEqual(
                model.calls[-1]["points"],
                [[[4.0, 4.0], [8.0, 4.0]]],
            )
            self.assertEqual(model.calls[-1]["labels"], [[1, 0]])
            self.assertGreater(expanded.mask.sum(), initial.mask.sum())
            self.assertLess(corrected.mask.sum(), initial.mask.sum())


if __name__ == "__main__":
    unittest.main()
