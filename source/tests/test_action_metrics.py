import unittest

import numpy as np

from starVLA.training.action_metrics import compute_action_metrics


class ActionMetricsTest(unittest.TestCase):
    def test_unmasked_metrics_are_per_element_means(self):
        predicted = np.array([[[1, -2], [3, -4]]], dtype=np.float32)
        result = compute_action_metrics(predicted, [{"action": np.zeros((2, 2))}])
        self.assertEqual(result, {"mae_score": 2.5, "mse_score": 7.5})

    def test_masked_metrics_use_only_valid_elements_across_batch(self):
        predicted = np.array([[[1, np.nan], [-3, np.inf]], [[2, 4], [np.nan, np.inf]]])
        examples = [
            {"action": [[0, np.nan], [0, np.inf]], "action_mask": [[1, 0], [1, 0]]},
            {"action": [[0, 0], [np.inf, np.nan]], "action_mask": [[1, 1], [0, 0]]},
        ]
        result = compute_action_metrics(predicted, examples)
        self.assertEqual(result, {"mae_score": 2.5, "mse_score": 7.5})

    def test_target_and_mask_use_final_prediction_horizon(self):
        examples = [{"action": [[100, 100], [1, 100], [3, 100]], "action_mask": [[1, 1], [1, 0], [1, 0]]}]
        result = compute_action_metrics(np.zeros((1, 2, 2)), examples)
        self.assertEqual(result, {"mae_score": 2.0, "mse_score": 5.0})

    def test_mixed_missing_masks_are_rejected(self):
        examples = [{"action": [[0]], "action_mask": [[1]]}, {"action": [[0]]}]
        with self.assertRaisesRegex(ValueError, "every sample"):
            compute_action_metrics(np.zeros((2, 1, 1)), examples)

    def test_all_invalid_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            compute_action_metrics(np.zeros((1, 1, 1)), [{"action": [[0]], "action_mask": [[0]]}])

    def test_no_valid_elements_in_predicted_horizon_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            compute_action_metrics(np.zeros((1, 1, 1)), [{"action": [[0], [0]], "action_mask": [[1], [0]]}])

    def test_shape_mismatches_are_rejected(self):
        for predicted, action in [
            (np.zeros((1, 1)), [[0]]),
            (np.zeros((2, 1, 1)), [[0]]),
            (np.zeros((1, 1, 2)), [[0]]),
            (np.zeros((1, 2, 1)), [[0]]),
            (np.zeros((1, 0, 1)), [[0]]),
        ]:
            with self.subTest(shape=predicted.shape), self.assertRaises(ValueError):
                compute_action_metrics(predicted, [{"action": action}])
        with self.assertRaisesRegex(ValueError, "action_mask shape"):
            compute_action_metrics(np.zeros((1, 1, 2)), [{"action": [[0, 0]], "action_mask": [1, 1]}])

    def test_nonbinary_or_nonfinite_masks_are_rejected(self):
        for mask in [np.nan, np.inf, -1, 0.5, 2]:
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "finite binary"):
                compute_action_metrics(np.zeros((1, 1, 1)), [{"action": [[0]], "action_mask": [[mask]]}])


if __name__ == "__main__":
    unittest.main()
