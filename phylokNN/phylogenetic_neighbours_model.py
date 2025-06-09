from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from phylokNN import get_first_column


class PhylNearestNeighbours(BaseEstimator):
    '''
    General idea
    # Given a phylogeny with traits to predict, convert to a distance matrix
    # Using the distance matrix, estimate unknown values using the following scheme:
    # For a given sample/species, return a weighted average of trait values of other species in the tree
    # where the weights are 1/distance ^ k.
    # This is equivalent to using KNNeighbours with K=N and weight=distance, with the edition of the kappa parameter.
    In cases where a species to predict is greater than the max distance to another labelled species, the mean trait value for the training species is used as the prediction.

    Assuming that the training is appropriate, this in part addresses issued raised in molinavenegas_how_2024 in a relatively computationally light
    manner by only using phylogenetic predictions where the distance is under some learnt threshold. Where the learnt threshold is 0, this indicates
    that taking the training mean is better.

    The class is set up unconventionally -- the fit and predict methods require X to be passed as a list of species names.
    Multidimensional X is not supported, a workaround could be to always use the first column of an array, this is not ideal but would allow proper incorporation with scikitlearn.
    '''

    def __init__(self, distance_matrix: pd.DataFrame, clf: bool, ratio_max_branch_length: float = None,
                 kappa: float = None, fill_in_unknowns_with_mean: bool = True):
        """
        :param distance_matrix: A pandas DataFrame representing the distance matrix between instances. Indices and columns should be taxon names. Should include all train, test species and unknown species for predictions.
        :param clf: A boolean indicating whether to output binary classes or continuous estimates for final predictions
        :param ratio_max_branch_length: A float value for the ratio of the largest tree distance to use as maximum distance threshold.
        :param kappa: A float values for Kappa used to modify branch lengths, similar to Pagel's Kappa frp, "‘Detecting Correlated Evolution on Phylogenies: A General Method for the Comparative Analysis of Discrete Characters’, Proceedings of the Royal Society of London. Series B: Biological Sciences 255, no. 1342 (22 January 1994): 37–45, https://doi.org/10.1098/rspb.1994.0006."

        """

        self.distance_matrix = distance_matrix  # .copy(deep=False) # Making a shallow copy to reduce RAM with multiple instances, but breaks when cloning class.
        self.clf = clf
        if ratio_max_branch_length is None:
            ratio_max_branch_length = 1
        self.ratio_max_branch_length = ratio_max_branch_length

        if kappa is None:
            kappa = 1
        self.kappa = kappa

        self.fill_in_unknowns_with_mean = fill_in_unknowns_with_mean

        if ratio_max_branch_length < 0 or ratio_max_branch_length > 1:
            raise ValueError('ratio_max_branch_length must be between 0 and 1')

    def __sklearn_tags__(self):
        # This isn't the best way to implement this, but is a quick fix.
        tags = super().__sklearn_tags__()
        if self.clf:
            tags.estimator_type = "classifier"
        return tags

    @staticmethod
    def check_integrity_of_distance_matrix(dist_matrix: pd.DataFrame):
        """
        Check the integrity of a distance matrix. The method does not return any value, it raises an AssertionError if there's an issue.

        :param dist_matrix: A pandas DataFrame representing a distance matrix.
        :return: None
        """
        # Check is symmetric
        pd.testing.assert_frame_equal(dist_matrix, dist_matrix.T)
        # Check columns=rows names
        assert dist_matrix.index.tolist() == dist_matrix.columns.tolist()
        # Check only zeros on diagonal
        diagonals = set(np.diag(dist_matrix.values))
        assert diagonals == {0}

        mask = ~np.eye(dist_matrix.values.shape[0], dtype=bool)
        non_diagonal_list = dist_matrix.values[mask].tolist()
        problems = {}
        for c in dist_matrix.columns:
            zero_rows = dist_matrix.index[dist_matrix[c] == 0].tolist()
            if len(zero_rows) > 1:
                problems[c] = zero_rows
        if len(problems) > 0:
            print(
                'Some taxa in the tree have zero distance. This may be a result of creating zero length tips from nodes which are the same MRCA for different sets of species.')
            print(problems)
        # assert 0 not in non_diagonal_list
        # Check no Nans
        assert not dist_matrix.isnull().values.any()

    @staticmethod
    def check_compatibility_of_matrix_and_data(dist_matrix: pd.DataFrame, target_df: pd.DataFrame):
        # Check names match in indices of target_df and dist_matrix
        overlapping_names = set(target_df.index).intersection(set(dist_matrix.index))
        if len(overlapping_names) == 0:
            raise AssertionError('No index names from target dataframe found in distance matrix indices.')

    @staticmethod
    def predict_phylogenetic_neighbours(dist_matrix: pd.DataFrame, train_plants: list, plants_to_predict: list, target_df: pd.DataFrame,
                                        target_name: str,
                                        kappa: float, max_distance: float, sample_weight=None, fill_in_unknowns_with_mean: bool = True):
        """

        :param fill_in_unknowns_with_mean:
        :param dist_matrix: The distance matrix containing phylogenetic distances between plants.
        :param train_plants: The list of plants used for 'training' the model.
        :param plants_to_predict: The list of plants for which to predict the phylogenetic neighbors.
        :param target_df: The DataFrame containing the target trait values for the plants.
        :param target_name: The name of the target trait.
        :param kappa: The exponent used to modify branch lengths/distances. Similar to Pagel's Kappa.
        :param max_distance: The maximum distance of neighbors to consider. If None, all neighbors are considered.
        :param sample_weight:
        :return: A DataFrame containing the predicted estimated values for the target trait for the given plants to predict.
        """
        problem_set = set(train_plants).intersection(set(plants_to_predict))
        if len(problem_set) > 0:
            print(
                f'WARNING: test plants found in train plants: {problem_set}. These will be removed from "train" plants during prediction on an individual basis.')
            # raise ValueError('Train and plants to predict not distinct.')

        ## Get data to use to predict
        training_target_df = target_df[target_df.index.isin(train_plants)]
        training_dist_matrix = dist_matrix[dist_matrix.index.isin(train_plants)]

        # Merge distances with trait value
        training_data = pd.merge(training_dist_matrix, training_target_df, left_index=True, right_index=True)
        if sample_weight is not None:
            sample_weight.name = 'sample_weight_'
            training_data = pd.merge(training_data, sample_weight, left_index=True, right_index=True)
        results = {}
        for sp in plants_to_predict:
            if sp in training_data.columns:
                if sample_weight is not None:
                    species_training_data = training_data[[sp, target_name, 'sample_weight_']].copy()
                else:
                    species_training_data = training_data[[sp, target_name]].copy()
                if len(problem_set) > 0:
                    species_training_data = species_training_data[species_training_data.index != sp]  # Remove this species if its in training data.
                # if len(species_training_data) == 0:
                #     raise ValueError(f'No relevant training distances found for {sp}. This is unexpected, indicating that the species is not in the tree, or that ')
                max_dist_df = species_training_data.copy()
                if max_distance is not None:
                    max_dist_df = max_dist_df[max_dist_df[sp] < max_distance]

                    # Where max distance is too restrictive, use closest other taxa
                    # if handle_distant_taxa == 'closest':
                    #
                    #     if len(max_dist_df) == 0:
                    #         closest_dist = relevant_df[sp].min()
                    #         max_dist_df = relevant_df[relevant_df[sp] <= closest_dist]
                if len(max_dist_df) > 0:
                    if sample_weight is not None:
                        max_dist_df['inverse_distance'] = max_dist_df['sample_weight_'] / (max_dist_df[sp] ** kappa)
                    else:
                        max_dist_df['inverse_distance'] = 1 / (max_dist_df[sp] ** kappa)

                    weighted_sum = (max_dist_df['inverse_distance'] * max_dist_df[target_name]).sum()
                    normalised_sum = weighted_sum / (max_dist_df['inverse_distance'].sum())
                    results[sp] = normalised_sum
                else:
                    if fill_in_unknowns_with_mean:
                        if sample_weight is not None:
                            results[sp] = (species_training_data[target_name] * species_training_data['sample_weight_']).sum() / \
                                          species_training_data[
                                              'sample_weight_'].sum()
                        else:
                            results[sp] = species_training_data[target_name].mean()
                    else:
                        results[sp] = np.nan
        out_df = pd.DataFrame.from_dict(results, orient='index', columns=['estimate'])

        return out_df

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight=None):
        """
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features). The first column must be species names.
        :param y: The target variable. An ArrayLike of shape (n_samples,).
        :param sample_weight: Optional. The sample weights. array-like of shape (n_samples,) default=None
        :return: None
        """
        X, y = validate_data(
            self,
            X,
            y, skip_check_array=True)

        self.target_name_ = '_unique_target_name__axmn'

        self.X_ = X

        self.train_plants_ = get_first_column(X)
        if isinstance(y, pd.Series):
            y_series = y
            assert self.train_plants_ == list(y_series.index)
        else:
            # Gridsearch Fit changes y to a numpy array so this won't work in that case
            y_series = pd.Series(y, index=self.train_plants_)

        if self.clf:
            self.classes_ = unique_labels(y_series.unique().tolist())

            if len(self.classes_) > 2:
                raise NotImplementedError('Method currently only implemented for binary classification. '
                                          'If your classes are ordinal, you could use the continuous version and round outputs.')
        self.labelled_training_data_ = y_series.to_frame(name=self.target_name_)

        assert len(X) == len(y)
        assert len(X) == len(y_series)
        self.mean_activity_ = np.average(y_series, weights=sample_weight)
        # print(f'Mean activity: {self.mean_activity_}')
        self.sample_weight_ = sample_weight
        if isinstance(self.sample_weight_, pd.Series):
            pd.testing.assert_index_equal(self.labelled_training_data_.index, self.sample_weight_.index)

        self.train_distances_ = self.distance_matrix[self.distance_matrix.index.isin(self.labelled_training_data_.index)]
        # Do some integrity check of the dist_matrix and target_df
        PhylNearestNeighbours.check_compatibility_of_matrix_and_data(self.distance_matrix, self.labelled_training_data_)

        max_distance_for_entire_training_tree = max(self.distance_matrix.select_dtypes(include=[np.number]).max())
        self.max_distance_ = max_distance_for_entire_training_tree * self.ratio_max_branch_length

        return self

    def _get_data_with_predictions(self, X_test: Union[pd.Series, list]) -> pd.DataFrame:
        '''
        Uses phylogenetic data and labelled data of species in self.train_plants, to predict APM activity of plants given in X.
        :param X_test:
        :return:
        '''
        check_is_fitted(self)
        if self.train_plants_ is None:
            raise ValueError('Not fitted')
        if isinstance(X_test, pd.Series):
            plants_to_predict = X_test.values.tolist()
        else:
            plants_to_predict = list(X_test.copy())
        overlap = set(plants_to_predict).intersection(set(self.train_plants_))
        if len(overlap) > 0:
            print(f'WARNING: test species found in train plants: {overlap}')

        # Get relevant distances to reduce memory
        relevant_train_plants = self.train_plants_
        relevant_test_plants = plants_to_predict
        distances = self.distance_matrix[self.distance_matrix.index.isin(relevant_train_plants + relevant_test_plants)]

        prediction_df = self.predict_phylogenetic_neighbours(distances, relevant_train_plants, relevant_test_plants, self.labelled_training_data_,
                                                             self.target_name_, sample_weight=self.sample_weight_,
                                                             kappa=self.kappa,
                                                             max_distance=self.max_distance_,
                                                             fill_in_unknowns_with_mean=self.fill_in_unknowns_with_mean)
        X = pd.Series(plants_to_predict, name='index_names')
        data_with_predictions = pd.merge(X, prediction_df, left_on='index_names',
                                         right_index=True,
                                         how='left')
        data_with_predictions = data_with_predictions.set_index('index_names', drop=True)
        if self.fill_in_unknowns_with_mean:
            self.fill_in_mean_activities(data_with_predictions)
        assert list(data_with_predictions.index) == plants_to_predict
        return data_with_predictions

    def predict(self, X: ArrayLike):
        """
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features). The first column must be species names.
        :return: ndarray of shape (n_samples,)
        """
        X = validate_data(self, X, reset=False, skip_check_array=True)
        names = get_first_column(X)
        data_with_predictions = self._get_data_with_predictions(names)
        if self.clf:
            threshold = .5
            data_with_predictions['state'] = np.where(
                data_with_predictions['estimate'].isna(), np.nan,
                data_with_predictions['estimate'].gt(threshold).astype(int))
            y_pred = data_with_predictions['state'].values.astype(float)
        else:
            y_pred = data_with_predictions['estimate'].values.astype(float)
        assert len(y_pred) == len(X)
        return y_pred

    def predict_proba(self, X: ArrayLike):
        """
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features). The first column must be species names.
        :return: array-like of shape (n_samples, n_classes)
        """
        if not self.clf:
            raise ValueError(
                f"This PhylNearestNeighbours instance is set with clf=False. Only classifiers should use predict_proba. Use predict() instead."
            )
        X = validate_data(self, X, reset=False, skip_check_array=True)
        names = get_first_column(X)
        data_with_predictions = self._get_data_with_predictions(names)
        data_with_predictions['0'] = 1 - data_with_predictions['estimate']
        T = np.column_stack((data_with_predictions['0'].values, data_with_predictions['estimate'].values)).astype(float)
        assert len(T) == len(X)

        return T

    def fill_in_mean_activities(self, prediction_df):
        # Assign mean values to unknown cases
        # This now happens in predict_phylogenetic_neighbours, but this may catch some edge cases.
        prediction_df['estimate'] = np.where(
            prediction_df['estimate'].isna(), self.mean_activity_,
            prediction_df['estimate'])


def get_gridsearch_best_hparams_for_phylnn(X_train: pd.DataFrame, y_train: pd.Series, distance_matrix: pd.DataFrame, clf: bool, cv,
                                           val_scorer: callable, greater_is_better: bool,
                                           kappas: list = None,
                                           ratios: list = None,
                                           sample_weight=None, fill_in_unknowns_with_mean: bool = True):
    """
    Does inner cross validation to return the best hyperparameters for the model.

    Maybe reuse/update this
    :param fill_in_unknowns_with_mean: whether to fill NaN predictions with mean during tuning.
    :param sample_weight:
    :param ratios:
    :param kappas:
    :param cv:
    :param clf:
    :param distance_matrix:
    :param y_train:
    :param X_train:
    :param val_scorer: The name of a the validation metric --- higher value is better!
        :param greater_is_better: Whether for val_scorer greater is better or not
    :return: None
    """
    relevant_species = list(set(X_train.index).intersection(set(distance_matrix.index)))
    distance_matrix_for_nested_cv = distance_matrix.loc[relevant_species, relevant_species]

    ratios = ratios or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    kappas = kappas or [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]

    best_score = None
    best_hyperparams = None

    splits = list(cv.split(X_train, y_train))  # Store as a list, as generators can only generate once.
    # print(splits)
    for kappa in kappas:
        for ratio in ratios:

            hparams = {
                'kappa': kappa,
                'ratio_max_branch_length': ratio
            }

            scores = []

            for i, (train_index, val_index) in enumerate(splits):

                X_inner_train, X_inner_validation = X_train.iloc[train_index], X_train.iloc[val_index]
                y_inner_train, y_inner_validation = y_train.iloc[train_index], y_train.iloc[val_index]

                if sample_weight is not None:
                    inner_train_weights = sample_weight.iloc[train_index]
                else:
                    inner_train_weights = None

                estim = PhylNearestNeighbours(distance_matrix_for_nested_cv, clf, ratio, kappa, fill_in_unknowns_with_mean=fill_in_unknowns_with_mean)
                estim.fit(X_inner_train.index, y_inner_train, sample_weight=inner_train_weights)

                # Note that this fills in nan values, so if the max distance is restrictive mean values will be output.
                prediction_df = estim._get_data_with_predictions(X_inner_validation.index)
                pd.testing.assert_index_equal(prediction_df.index, X_inner_validation.index, check_names=False)
                pd.testing.assert_index_equal(prediction_df.index, y_inner_validation.index, check_names=False)

                # Remove nan predictions
                prediction_df = prediction_df.dropna(subset=["estimate"])
                if len(prediction_df) > 0:
                    if sample_weight is not None:
                        inner_val_weights = sample_weight[sample_weight.index.isin(prediction_df.index)]
                    else:
                        inner_val_weights = None
                    y_inner_validation = y_inner_validation[y_inner_validation.index.isin(prediction_df.index)]
                    if len(y_inner_validation) > 0:
                        val_score = val_scorer(y_inner_validation, prediction_df['estimate'].values,
                                               sample_weight=inner_val_weights)
                        if val_score < 0:
                            raise ValueError('Scorer must return values >= 0.')
                        scores.append(val_score)
                    else:
                        raise Exception('Non nan predictions found, but no validation labels.')

            if len(scores) > 0:
                mean_val_score = np.mean(scores)
                if (best_score is None) or (greater_is_better and mean_val_score > best_score) or (
                        not greater_is_better and mean_val_score < best_score):
                    best_score = mean_val_score
                    best_hyperparams = hparams
            else:
                print(f'WARNING: no successful splits found for kappa: {kappa} and ratio: {ratio}')

    if best_score is None:
        raise Exception('No successful splits with non NaN predictions found for any kappa or ratio.')
    print(f'Best neighbour parameters: {best_hyperparams} with score: {best_score}')
    if best_hyperparams['ratio_max_branch_length'] == 0:
        print(
            'WARNING: optimal ratio_max_branch_length is set to 0, this may indicate that the tree is not providing useful predictions and simply taking the mean value is better.')
    return best_hyperparams
