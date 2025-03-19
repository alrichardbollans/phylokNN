import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score

sklearn.set_config(enable_metadata_routing=True)

from phyloKNN import PhylNearestNeighbours


def phyloNN_gridsearch(distance_matrix: pd.DataFrame, clf: bool, scorer, cv, X, y, weights=None,
                       ratios: list = None, kappas: list = None, njobs=-1):
    '''
    A utility functin showing how best to use gridsearch with phyloKNN.
    :param distance_matrix:
    :param clf:
    :param scorer:
    :param cv:
    :param X:
    :param y:
    :param weights:
    :param ratios:
    :param kappas:
    :param njobs:
    :return:
    '''

    if kappas is None:
        kappas = [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]
    if ratios is None:
        ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    phyln = PhylNearestNeighbours(distance_matrix, clf, 1, 1, fill_in_unknowns_with_mean=False)

    gs = GridSearchCV(
        estimator=phyln,
        param_grid={'ratio_max_branch_length': ratios,
                    'kappa': kappas},
        cv=cv,
        n_jobs=njobs,
        scoring=scorer,
        verbose=1,
        error_score="raise",
        refit=True
    )

    phyln.set_fit_request(sample_weight=True)
    fitted_gs = gs.fit(X, y, sample_weight=weights)
    print(fitted_gs.best_params_)
    if fitted_gs.best_params_['ratio_max_branch_length'] == 0:
        print(
            f'WARNING: Max distance set to 0, this means unweighted means performed best in gridsearch and '
            f'that NaNs/mean values will be predicted for all inputs (barring polytomies).')
    return fitted_gs


def phyloNN_bayes_opt(distance_matrix: pd.DataFrame, clf: bool, scorer, cv, X:pd.DataFrame, y, weights=None, njobs=1, verbose=2, init_points=10, n_iter=50):
    '''
    Example of how to use bayesian optimisation with phyloKNN.
    :param distance_matrix:
    :param clf:
    :param scorer:
    :param cv:
    :param X:
    :param y:
    :param weights:
    :param njobs:
    :param verbose: param for BayesianOptimization
    :param init_points: param for BayesianOptimization
    :param n_iter: param for BayesianOptimization
    :return:
    '''

    if clf:
        if scorer._response_method == 'predict':
            print(f'WARNING: reponse method "predict" for scoring classifier with {scorer}')

    assert len(set(distance_matrix.columns).intersection(set(X[X.columns[0]].values))) > 0

    from bayes_opt import BayesianOptimization

    global _worst_score
    _worst_score = None
    def black_box_function(ratio, kappa):
        """Function with unknown internals we wish to maximize.

        When NaN is returned for all values of cross_val_score, the current worst score is returned. This will break if the first try returns NaN.
        """

        if distance_matrix is None:
            return 0
        phyln = PhylNearestNeighbours(distance_matrix, clf, ratio, kappa, fill_in_unknowns_with_mean=False)
        # Add metadating routing for sample weights
        scorer.set_score_request(sample_weight=True)
        phyln.set_fit_request(sample_weight=True)
        cv_score = cross_val_score(phyln, X, y, cv=cv, scoring=scorer, n_jobs=njobs, params={'sample_weight': weights}, error_score="raise")
        out = np.mean(cv_score)
        global _worst_score
        if np.isnan(out):
            return _worst_score
        elif _worst_score is None:
            _worst_score = out
        elif out < _worst_score:
            _worst_score = out

        return out

    # Bounded region of parameter space
    pbounds = {'ratio': (0, 1), 'kappa': (0, 3)}

    while True:

        try:
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=None,
                verbose=verbose
            )

            optimizer.maximize(init_points=init_points, n_iter=n_iter)
            break
        except TypeError:
            print(f'WARNING: Bayesian optimization failed to initialise -- retrying..')

    print(optimizer.max)
    best_ratio = optimizer.max['params']['ratio']
    best_kappa = optimizer.max['params']['kappa']
    if best_ratio < 0.05:
        print(
            f'WARNING: Max distance set to a small ratio: {best_ratio}, this may mean unweighted means performed best in hyperparameter search and '
            f'that NaNs/mean values will be predicted for all inputs (barring polytomies).')
    if best_kappa < 0.05:
        print(
            f'WARNING: kappa set to small value: {best_kappa}, this may mean unweighted means performed best in hyperparameter search and '
            f'that distances are not useful in predictions.')

    return best_ratio, best_kappa
