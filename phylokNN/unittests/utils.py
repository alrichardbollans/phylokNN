import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, accuracy_score, mean_squared_error

from phylokNN import get_first_column, nan_safe_metric_wrapper


class test_arrays(unittest.TestCase):


    def test_types(self):
        X = [['A', 2, 3], ['B', 5, 6], ['C', 8, 9]]

        assert get_first_column(X) == ['A', 'B', 'C']

        df = pd.DataFrame(X)
        assert get_first_column(df) == ['A', 'B', 'C']

        num = np.array(X)
        assert get_first_column(num) == ['A', 'B', 'C']

        self.assertRaises(AssertionError, get_first_column, [[1, 2, 3], ['B', 5, 6], ['C', 8, 9]])


class TestNanSafeMetric(unittest.TestCase):
    def setUp(self):
        # Define test data
        self.y_true = np.array([1, 0, 1, 0, 1])
        self.y_pred_with_nans = np.array([0.9, np.nan, 0.8, 0.3, np.nan])
        self.y_pred_no_nans = np.array([0.9, 0.2, 0.8, 0.3, 0.7])

    def test_brier_score_loss_with_nans(self):
        # Wrap brier_score_loss to handle NaNs
        brier_score_loss_nan_safe = nan_safe_metric_wrapper(brier_score_loss)

        # Calculate the expected result by manually filtering NaNs
        mask = ~np.isnan(self.y_pred_with_nans)
        y_true_filtered = self.y_true[mask]
        y_pred_filtered = self.y_pred_with_nans[mask]
        expected = brier_score_loss(y_true_filtered, y_pred_filtered)

        # Calculate the actual result using the wrapped function
        actual = brier_score_loss_nan_safe(self.y_true, self.y_pred_with_nans)

        # Assert that the results match
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(actual, brier_score_loss([1,1,0],[0.9,0.8,0.3]))

    def test_brier_score_loss_no_nans(self):
        # Wrap brier_score_loss to handle NaNs
        brier_score_loss_nan_safe = nan_safe_metric_wrapper(brier_score_loss)

        # Calculate the expected result (no NaNs to filter)
        expected = brier_score_loss(self.y_true, self.y_pred_no_nans)

        # Calculate the actual result using the wrapped function
        actual = brier_score_loss_nan_safe(self.y_true, self.y_pred_no_nans)

        # Assert that the results match
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(actual, brier_score_loss([1, 0, 1, 0, 1],[0.9, 0.2, 0.8, 0.3, 0.7]))


    def test_accuracy_score_with_nans(self):
        # Wrap accuracy_score to handle NaNs
        accuracy_score_nan_safe = nan_safe_metric_wrapper(accuracy_score)

        # Convert probabilities to class predictions for accuracy_score
        y_pred_classes_with_nans =  np.array([1, np.nan, 1, 0, np.nan])

        # Calculate the expected result by manually filtering NaNs
        mask = ~np.isnan(y_pred_classes_with_nans)
        y_true_filtered = self.y_true[mask]
        y_pred_filtered = y_pred_classes_with_nans[mask]
        expected = accuracy_score(y_true_filtered, y_pred_filtered)

        # Calculate the actual result using the wrapped function
        actual = accuracy_score_nan_safe(self.y_true, y_pred_classes_with_nans)

        # Assert that the results match
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(actual, accuracy_score([1,1,0],[1,1,0]))


    def test_mean_squared_error_with_nans(self):
        # Wrap mean_squared_error to handle NaNs
        mse_nan_safe = nan_safe_metric_wrapper(mean_squared_error)

        # Calculate the expected result by manually filtering NaNs
        mask = ~np.isnan(self.y_pred_with_nans)
        y_true_filtered = self.y_true[mask]
        y_pred_filtered = self.y_pred_with_nans[mask]
        expected = mean_squared_error(y_true_filtered, y_pred_filtered)

        # Calculate the actual result using the wrapped function
        actual = mse_nan_safe(self.y_true, self.y_pred_with_nans)

        # Assert that the results match
        self.assertAlmostEqual(actual, expected, places=6)

    def test_all_nans(self):
        # Test case where all predictions are NaNs
        y_true = np.array([1, 0, 1])
        y_pred_all_nans = np.array([np.nan, np.nan, np.nan])

        # Wrap brier_score_loss to handle NaNs
        brier_score_loss_nan_safe = nan_safe_metric_wrapper(brier_score_loss)


        self.assertTrue(np.isnan(brier_score_loss_nan_safe(y_true, y_pred_all_nans)))


if __name__ == '__main__':
    unittest.main()
