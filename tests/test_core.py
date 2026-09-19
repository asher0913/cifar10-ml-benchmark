import unittest

import numpy as np

from src.data import _flatten_and_normalize, _take_subset
from src.metrics import classification_metrics


class CorePipelineTests(unittest.TestCase):
    def test_image_flattening_normalizes_pixels(self):
        images = np.array([[[[0, 127, 255], [255, 0, 127]]]], dtype=np.uint8)
        flattened = _flatten_and_normalize(images)

        self.assertEqual(flattened.shape, (1, 6))
        self.assertAlmostEqual(float(flattened.min()), 0.0)
        self.assertAlmostEqual(float(flattened.max()), 1.0)

    def test_subsampling_is_reproducible(self):
        features = np.arange(100).reshape(20, 5)
        labels = np.arange(20)
        first = _take_subset(features, labels, subset=7, seed=42)
        second = _take_subset(features, labels, subset=7, seed=42)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_metrics_include_requested_classes(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            class_names=["cat", "dog"],
            label_indices=[0, 1],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(set(metrics["per_class"]), {"cat", "dog"})


if __name__ == "__main__":
    unittest.main()
