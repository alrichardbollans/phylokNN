import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

from phyloNN import PhylNearestNeighbours, get_gridsearch_best_hparams_for_phylnn


class Testpredict_phylogenetic_neighbours_with_all_neighbours(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Set up some mock data to be used across all the tests
        cls.distance_matrix = pd.DataFrame({
            'plantA': [0, 1.1, 1.3],
            'plantB': [1.1, 0, 1.5],
            'plantC': [1.3, 1.5, 0]},
            index=['plantA', 'plantB', 'plantC'])

        cls.target_df = pd.DataFrame({
            'Trait1': [0.5, 1.2, 0.8]},
            index=['plantA', 'plantB', 'plantC'])

    def test_predict_phylogenetic_neighbours_with_all_neighbours(self):
        # Given
        train_plants = ['plantA', 'plantB']
        plants_to_predict = ['plantC']
        target_name = 'Trait1'
        kappa = 1
        k_ratio = None  # this means all neighbours are considered

        # When
        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, kappa, k_ratio)

        # Then
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['estimate'].iloc[0], 0.825)

    def test_predict_phylogenetic_neighbours_with_distance(self):
        # Given
        train_plants = ['plantA', 'plantB']
        plants_to_predict = ['plantC']
        target_name = 'Trait1'
        kappa = 1
        max_distance = 1.5

        # When
        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, 0.5, max_distance=None)

        # Then
        self.assertIsInstance(result, pd.DataFrame)
        self.assertAlmostEqual(result['estimate'].iloc[0], 0.837484, places=5)

        # When
        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, kappa, max_distance)

        # Then
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['estimate'].iloc[0], 0.5)

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, kappa, 0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    def test_outside_threshold_distance(self):
        train_plants = ['plantA', 'plantB']
        plants_to_predict = ['plantC']
        target_name = 'Trait1'
        kappa = 1

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, kappa, 0)
        self.assertEqual(result['estimate'].iloc[0], 0.85)

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, kappa, 1)
        self.assertEqual(result['estimate'].iloc[0], 0.85)

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, ['plantA'], plants_to_predict, self.target_df, target_name, kappa, 1)
        self.assertEqual(result['estimate'].iloc[0], 0.5)

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, ['plantA', 'plantC'], ['plantB'], self.target_df, target_name, kappa, 1)
        self.assertEqual(result['estimate'].iloc[0], 0.65)

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            self.distance_matrix, ['plantA', 'plantC'], ['plantB'], self.target_df, target_name, -1, 1)
        self.assertEqual(result['estimate'].iloc[0], 0.65)

    def test_unknown_plants(self):
        train_plants = ['plantA', 'plantB', 'plantC']
        plants_to_predict = ['plantD']
        target_name = 'Trait1'
        kappa = 1
        distance_matrix = pd.DataFrame({
            'plantA': [0, 1.1, 1.3, 4.3],
            'plantB': [1.1, 0, 1.5, 3],
            'plantC': [1.3, 1.5, 0, 2],
            'plantD': [4.3, 3, 2, 0]
        },
            index=['plantA', 'plantB', 'plantC', 'plantD'])
        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            distance_matrix, train_plants, plants_to_predict, self.target_df, target_name, kappa, None)
        self.assertEqual(round(result['estimate'].iloc[0], 6), 0.859636)

        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(
            distance_matrix, train_plants, ['plantE'], self.target_df, target_name, kappa, None)
        self.assertEqual(len(result), 0)


class TestPhylNearestNeighbours(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.distance_matrix = pd.DataFrame([[0, 1, 3, 2, 6],
                                            [2, 2, 5, 7, 2],
                                            [2, 4, 2, 1, 0.5],
                                            [3, 5, 7, 2, 1],
                                            [3, 1, 3, 2, 9]]
                                           )
        cls.distance_matrix.columns = ['A', 'B', 'C', 'D', 'E']
        cls.distance_matrix.index = ['A', 'B', 'C', 'D', 'E']
        cls.target_df = pd.DataFrame({'target': [1, 0, 1, 1, 0]}, index=['A', 'B', 'C', 'D', 'E'])
        cls.clf = PhylNearestNeighbours(cls.distance_matrix, clf=True, ratio_max_branch_length=0.6, kappa=0.1)

    def test_init(self):
        pd.testing.assert_frame_equal(self.clf.distance_matrix, self.distance_matrix)

    def test_check_integrity_of_distance_matrix(self):
        with self.assertRaises(AssertionError):
            self.clf.check_integrity_of_distance_matrix(pd.DataFrame(np.random.rand(2, 3)))

    def test_check_compatibility_of_matrix_and_data(self):
        self.clf.check_compatibility_of_matrix_and_data(self.distance_matrix,
                                                        pd.DataFrame({'target': [1, 2, 3, np.nan]}, index=['A', 'B', 'C', 'E']))
        with self.assertRaises(AssertionError):
            self.clf.check_compatibility_of_matrix_and_data(self.distance_matrix,
                                                            pd.DataFrame({'target': [1, 2, 3, np.nan]}, index=['a', 'b', 'c', 'd']))

    def test_predict_phylogenetic_neighbours(self):
        result = PhylNearestNeighbours.predict_phylogenetic_neighbours(self.distance_matrix,
                                                                       ['A', 'B'], ['C', 'D'], self.target_df, 'target', 0.2, None)
        self.assertTrue(isinstance(result, pd.DataFrame), "Should return Pandas DataFrame")

    def test_fit(self):
        X = ['A', 'B', 'C']
        y = self.target_df.loc[['A', 'B', 'C']]['target']
        clf = self.clf.fit(X, y)
        self.assertTrue(hasattr(clf, 'train_plants_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'X_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'target_name'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'classes_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'labelled_training_data'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'mean_activity'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'sample_weight'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'train_distances'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'max_distance'), 'Model is not fitted correctly')

    def test_predict(self):
        X = self.distance_matrix.loc[['D', 'E']]
        y = self.target_df.loc[['D', 'E']]['target']
        clf = self.clf
        clf.fit(pd.Series(['A', 'B', 'C']), self.target_df.loc[['A', 'B', 'C']]['target'])
        predictions = clf.predict(['D', 'E'])
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict method should return Pandas Series.')
        self.assertTrue(np.array_equal(predictions, [1, 1]), True)

    def test__get_data_with_predictions(self):
        clf = self.clf
        clf.fit(self.distance_matrix.loc[['A', 'B', 'C']].index, self.target_df.loc[['A', 'B', 'C']]['target'])
        result = clf._get_data_with_predictions(self.distance_matrix.loc[['D', 'E']].index)
        assert result.shape == (2, 1)
        assert result.index.equals(pd.Index(['D', 'E']))
        assert result.loc['D', 'estimate'] == 1
        assert round(result.loc['E', 'estimate'],4) == 0.5346

        nan_result = clf._get_data_with_predictions(['X', 'Y'], fill_in_unknowns_with_mean=False)
        self.assertTrue(np.isnan(nan_result.loc['X', 'estimate']))
        self.assertTrue(np.isnan(nan_result.loc['Y', 'estimate']))

    def test_predict_proba(self):
        X = self.distance_matrix.loc[['D', 'E']]
        y = self.target_df.loc[['D', 'E']]['target']
        clf = self.clf
        clf.fit(pd.Series(['A', 'B', 'C']), self.target_df.loc[['A', 'B', 'C']]['target'])
        predictions = clf.predict_proba(['D', 'E'])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.5346], decimal=4)

    def test_fill_in_mean_activities(self):
        X = ['A', 'B', 'C', 'D', 'E']
        y = self.target_df['target']
        df = pd.DataFrame({'estimate': [1, np.nan, 3, 4]})
        clf = self.clf
        clf.fit(X, y)
        expected_df = pd.DataFrame({'estimate': [1, 0.6, 3, 4]})
        clf.fill_in_mean_activities(df)
        pd.testing.assert_frame_equal(df, expected_df)

    def test_fit_sample_weights(self):
        weights = pd.Series([1, 3, 9], index=['A', 'B', 'C'])
        X = self.distance_matrix.loc[['D', 'E']]
        y = self.target_df.loc[['D', 'E']]['target']
        clf = self.clf
        clf.fit(['A', 'B', 'C'], self.target_df.loc[['A', 'B', 'C']]['target'], sample_weight=weights)
        predictions = clf.predict(pd.Series(['D', 'E']))
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict method should return Pandas Series.')
        self.assertTrue(np.array_equal(predictions, [1, 1]), True)

        predictions = clf.predict_proba(['D', 'E'])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.7751], decimal=4)

    def test_sklearn_checks(self):
        check_estimator(self.clf)


class testgridsearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Set up some mock data to be used across all the tests
        np.random.seed(42)
        cls.distance_matrix = pd.DataFrame([[0, 1, 3, 2, 6],
                                            [2, 2, 5, 7, 2],
                                            [2, 4, 2, 1, 0.5],
                                            [3, 5, 7, 2, 1],
                                            [3, 1, 3, 2, 9]]
                                           )
        cls.distance_matrix.columns = ['A', 'B', 'C', 'D', 'E']
        cls.distance_matrix.index = ['A', 'B', 'C', 'D', 'E']
        cls.target_df = pd.DataFrame({'target': [1, 0, 1, 1, 0]}, index=['A', 'B', 'C', 'D', 'E'])

    def test_get_gridsearch_best_hparams_for_phylnn(self):
        cv = KFold(n_splits=2, shuffle=True, random_state=3)
        val_scorer = mean_absolute_error
        best_hparams = get_gridsearch_best_hparams_for_phylnn(self.distance_matrix.loc[['A', 'B', 'C']],
                                                              self.target_df.loc[['A', 'B', 'C']]['target'],
                                                              self.distance_matrix, clf=True, cv=cv,
                                                              val_scorer=val_scorer, greater_is_better=False)

        phyln = PhylNearestNeighbours(self.distance_matrix, clf=True, ratio_max_branch_length=best_hparams['ratio_max_branch_length'],
                                      kappa=best_hparams['kappa'])
        phyln.fit(self.distance_matrix.loc[['A', 'B', 'C']].index, self.target_df.loc[['A', 'B', 'C']]['target'])

        predictions = phyln.predict_proba(['D', 'E'])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.6716], decimal=4)

    def test_get_gridsearch_best_hparams_for_phylnn_with_SW(self):
        weights = pd.Series([1, 3, 9], index=['A', 'B', 'C'])
        cv = KFold(n_splits=2, shuffle=True, random_state=3)
        val_scorer = mean_absolute_error
        best_hparams = get_gridsearch_best_hparams_for_phylnn(self.distance_matrix.loc[['A', 'B', 'C']],
                                                              self.target_df.loc[['A', 'B', 'C']]['target'],
                                                              self.distance_matrix, clf=True, cv=cv,
                                                              val_scorer=val_scorer, greater_is_better=False, sample_weight=weights)

        phyln = PhylNearestNeighbours(self.distance_matrix, clf=True, ratio_max_branch_length=best_hparams['ratio_max_branch_length'],
                                      kappa=best_hparams['kappa'])
        phyln.fit(self.distance_matrix.loc[['A', 'B', 'C']].index, self.target_df.loc[['A', 'B', 'C']]['target'], sample_weight=weights)
        X_test = self.distance_matrix.loc[['D', 'E']]
        predictions = phyln.predict_proba(X_test.index)[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.7892], decimal=4)

    def test_try_proper_gridsearch(self):
        phyln = PhylNearestNeighbours(self.distance_matrix, True, 0, 0)
        gs = GridSearchCV(
            estimator=phyln,
            param_grid={'ratio_max_branch_length': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'kappa': [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3, 4]},
            cv=KFold(n_splits=2, shuffle=True, random_state=1),
            n_jobs=1,
            scoring=mean_absolute_error,
            verbose=1,
            error_score='raise',
            refit=True
        )

        fitted_gs = gs.fit(['A', 'B', 'C'], self.target_df.loc[['A', 'B', 'C']]['target'].values)

        print(self.name)
        print(fitted_gs.best_params_)


if __name__ == '__main__':
    unittest.main()
