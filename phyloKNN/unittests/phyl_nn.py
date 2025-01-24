import unittest

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, make_scorer, brier_score_loss
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

from phyloKNN import PhylNearestNeighbours, get_gridsearch_best_hparams_for_phylnn, phyloNN_gridsearch, phyloNN_bayes_opt, nan_safe_metric_wrapper


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
        cls.target_df = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 'target': [1, 0, 1, 1, 0]}, index=['A', 'B', 'C', 'D', 'E'])
        cls.clf = PhylNearestNeighbours(cls.distance_matrix, clf=True, ratio_max_branch_length=0.6, kappa=0.1)
        cls.reg = PhylNearestNeighbours(cls.distance_matrix, clf=False, ratio_max_branch_length=0.6, kappa=0.1)

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
        X = [['A'], ['B'], ['C']]
        y = self.target_df.loc[['A', 'B', 'C']]['target']
        clf = self.clf.fit(X, y)
        self.assertTrue(hasattr(clf, 'train_plants_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'X_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'target_name_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'classes_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'labelled_training_data_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'mean_activity_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'sample_weight_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'train_distances_'), 'Model is not fitted correctly')
        self.assertTrue(hasattr(clf, 'max_distance_'), 'Model is not fitted correctly')

    def test_predict(self):
        X = self.distance_matrix.loc[['D', 'E']]
        y = self.target_df.loc[['D', 'E']]['target']
        clf = self.clf
        clf.fit(self.target_df.loc[['A', 'B', 'C']], self.target_df.loc[['A', 'B', 'C']]['target'])
        predictions = clf.predict([['D', 9], ['E', 11]])
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict method should return Pandas Series.')
        self.assertTrue(np.array_equal(predictions, [1, 1]), True)

    def test__get_data_with_predictions(self):
        clf = self.clf
        clf.fit(self.target_df.loc[['A', 'B', 'C']], self.target_df.loc[['A', 'B', 'C']]['target'])
        result = clf._get_data_with_predictions(self.distance_matrix.loc[['D', 'E']].index)
        assert result.shape == (2, 1)
        assert result.index.equals(pd.Index(['D', 'E']))
        assert result.loc['D', 'estimate'] == 1
        assert round(result.loc['E', 'estimate'], 4) == 0.5346

        clf.fill_in_unknowns_with_mean = False
        nan_result = clf._get_data_with_predictions(['X', 'Y'])
        self.assertTrue(np.isnan(nan_result.loc['X', 'estimate']))
        self.assertTrue(np.isnan(nan_result.loc['Y', 'estimate']))

    def test_predict_proba(self):
        X = self.distance_matrix.loc[['D', 'E']]
        y = self.target_df.loc[['D', 'E']]['target']
        clf = self.clf
        clf.fit(self.target_df.loc[['A', 'B', 'C']], self.target_df.loc[['A', 'B', 'C']]['target'])
        predictions = clf.predict_proba([['D', 9], ['E', 9]])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.5346], decimal=4)

        self.reg.fit(self.target_df.loc[['A', 'B', 'C']], self.target_df.loc[['A', 'B', 'C']]['target'])
        self.assertRaises(ValueError, self.reg.predict_proba, [['D', 9], ['E', 9]])

    def test_fill_in_mean_activities(self):
        X = [['A'], ['B'], ['C'], ['D'], ['E']]
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
        clf.fit([['A'], ['B'], ['C']], self.target_df.loc[['A', 'B', 'C']]['target'], sample_weight=weights)
        predictions = clf.predict([['D'], ['E']])
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict method should return Pandas Series.')
        self.assertTrue(np.array_equal(predictions, [1, 1]), True)

        predictions = clf.predict_proba([['D'], ['E']])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.7751], decimal=4)

    def test_predictions_with_nans(self):
        # When ratio max dist is 0, should get nan predictions if fill_in_unknowns_with_mean=False
        phyln = PhylNearestNeighbours(self.distance_matrix, True, 0, 0, fill_in_unknowns_with_mean=False)
        self.assertRaises(sklearn.exceptions.NotFittedError, phyln.predict, [['D', 9], ['E', 9]])

        phyln.fit(self.target_df.loc[['A', 'B', 'C']], self.target_df.loc[['A', 'B', 'C']]['target'])
        x = phyln.predict([['D', 9], ['E', 9]])
        np.testing.assert_array_equal(x, [np.nan, np.nan])

        x = phyln.predict([['D', 9], ['A', 9]])
        np.testing.assert_array_equal(x, [np.nan, np.nan])

        phyln = PhylNearestNeighbours(self.distance_matrix, True, 0, 0, fill_in_unknowns_with_mean=False)
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        cv = KFold(n_splits=2, shuffle=True, random_state=3)
        gs = GridSearchCV(
            estimator=phyln,
            param_grid={'ratio_max_branch_length': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'kappa': [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]},
            cv=cv,
            n_jobs=1,
            scoring=mae_scorer,
            verbose=1,
            error_score='raise',
            refit=True
        )
        self.assertRaises(ValueError, gs.fit, [['A'], ['B'], ['C']], [1, 0, 1])

    def test_raises_correct_errors(self):
        phyln = PhylNearestNeighbours(self.distance_matrix, False, 0, 0, fill_in_unknowns_with_mean=False)
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        cv = KFold(n_splits=2, shuffle=True, random_state=3)
        gs = GridSearchCV(
            estimator=phyln,
            param_grid={'ratio_max_branch_length': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'kappa': [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]},
            cv=cv,
            n_jobs=1,
            scoring=mae_scorer,
            verbose=1,
            error_score='raise',
            refit=True
        )
        # Should raise a Value error because of the 0 ratio means nans are predicted when scoring
        self.assertRaises(ValueError, gs.fit, [['A'], ['B'], ['C']], [1, 0, 1])

        phyln.fit(self.target_df.loc[['A', 'B', 'C']], self.target_df.loc[['A', 'B', 'C']]['target'])

        x = phyln.predict([['D', 9], ['E', 9]])
        np.testing.assert_array_equal(x, [np.nan, np.nan])

        x = phyln.predict([['D', 9], ['A', 9]])
        np.testing.assert_array_equal(x, [np.nan, np.nan])

    def test_sklearn_checks(self):
        check_estimator(self.clf, on_fail='warn')

        # This will fail becasue we force first column to be string names
        check_estimator(self.clf)  # , on_fail='warn')


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

    @unittest.skip('No longer used')
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

    @unittest.skip('No longer used')
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
        # This is how to do gridsearch -- set fill in means to off with error_score = np.nan and then fit a new model.
        phyln = PhylNearestNeighbours(self.distance_matrix, True, 1, 1, fill_in_unknowns_with_mean=False)
        mean_absolute_error_nan_safe = nan_safe_metric_wrapper(mean_absolute_error)

        mae_scorer = make_scorer(mean_absolute_error_nan_safe, greater_is_better=False)
        cv = KFold(n_splits=2, shuffle=True, random_state=3)
        gs = GridSearchCV(
            estimator=phyln,
            param_grid={'ratio_max_branch_length': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'kappa': [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]},
            cv=cv,
            n_jobs=1,
            scoring=mae_scorer,
            verbose=1,
            error_score='raise',
            refit=False
        )

        fitted_gs = gs.fit([['A'], ['B'], ['C']], [1, 0, 1])
        print(fitted_gs.best_params_)
        best_phyln = PhylNearestNeighbours(self.distance_matrix, True, fitted_gs.best_params_['ratio_max_branch_length'],
                                           fitted_gs.best_params_['kappa'], fill_in_unknowns_with_mean=True)
        best_phyln.fit([['A'], ['B'], ['C']], [1, 0, 1])
        predictions = best_phyln.predict_proba([['D'], ['E']])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.5346], decimal=4)

    def test_try_proper_gridsearch_with_SW(self):
        weights = pd.Series([1, 3, 9], index=['A', 'B', 'C'])

        # This is how to do gridsearch -- set fill in means to off with error_score = np.nan and then fit a new model.
        phyln = PhylNearestNeighbours(self.distance_matrix, True, 1, 1, fill_in_unknowns_with_mean=False)
        mean_absolute_error_nan_safe = nan_safe_metric_wrapper(mean_absolute_error)
        mae_scorer = make_scorer(mean_absolute_error_nan_safe, greater_is_better=False)
        cv = KFold(n_splits=2, shuffle=True, random_state=3)
        gs = GridSearchCV(
            estimator=phyln,
            param_grid={'ratio_max_branch_length': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        'kappa': [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]},
            cv=cv,
            n_jobs=1,
            scoring=mae_scorer,
            verbose=1,
            error_score='raise',
            refit=False
        )
        phyln.set_fit_request(sample_weight=True)

        fitted_gs = gs.fit([['A'], ['B'], ['C']], [1, 0, 1], sample_weight=weights)
        print(fitted_gs.best_params_)
        if fitted_gs.best_params_['ratio_max_branch_length'] == 0:
            print(
                f'WARNING: Max distance set to 0, this means unweighted means performed best in gridsearch and that NaNs/mean values will be predicted for all inputs (barring polytomies).')
        best_phyln = PhylNearestNeighbours(self.distance_matrix, True, fitted_gs.best_params_['ratio_max_branch_length'],
                                           fitted_gs.best_params_['kappa'], fill_in_unknowns_with_mean=False)
        best_phyln.fit([['A'], ['B'], ['C']], [1, 0, 1], sample_weight=weights)
        predictions = best_phyln.predict_proba([['D'], ['E']])[:, 1]
        self.assertTrue(isinstance(predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(predictions, [1, 0.7751], decimal=4)

        ## my gs example

        gs = phyloNN_gridsearch(self.distance_matrix, clf=True, scorer=mae_scorer, cv=cv, X=[['A'], ['B'], ['C']], y=[1, 0, 1], weights=weights)
        gs_predictions = gs.predict_proba([['D'], ['E']])[:, 1]
        self.assertTrue(isinstance(gs_predictions, np.ndarray), 'Predict_proba method should return Pandas DataFrame.')
        np.testing.assert_array_almost_equal(gs_predictions, [1, 0.7751], decimal=4)

    def test_bayes(self):

        brier_score_loss_nan_safe = nan_safe_metric_wrapper(brier_score_loss)

        _scorer = make_scorer(brier_score_loss_nan_safe, greater_is_better=False)
        r,k = phyloNN_bayes_opt(self.distance_matrix, clf=True, scorer=_scorer, cv=KFold(n_splits=2, shuffle=True, random_state=3),
                                      X=[['A'], ['B'], ['C']], y=[1, 0, 1], init_points=10, n_iter=10)

        _scorer = make_scorer(brier_score_loss_nan_safe, greater_is_better=False)
        weights = pd.Series([1, 3, 9], index=['A', 'B', 'C'])
        r2,k2 = phyloNN_bayes_opt(self.distance_matrix, clf=True, scorer=_scorer, cv=KFold(n_splits=2, shuffle=True, random_state=3),
                                      X=[['A'], ['B'], ['C']], y=[1, 2, 1], weights=weights, init_points=10, n_iter=10)

        print(r)
        print(r2)
        print(k2)
        assert (r != r2) or (k != k2)


if __name__ == '__main__':
    unittest.main()
