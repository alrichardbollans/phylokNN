import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

from phyloKNN import PhylNearestNeighbours


def gridsearch(distance_matrix: pd.DataFrame, clf: bool, scorer, cv, X, y, weights=None,
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
    phyln = PhylNearestNeighbours(distance_matrix, clf, 1, 1, fill_in_unknowns_with_mean=True)
    gs = GridSearchCV(
        estimator=phyln,
        param_grid={'ratio_max_branch_length': ratios,
                    'kappa': kappas},
        cv=cv,
        n_jobs=njobs,
        scoring=scorer,
        verbose=1,
        error_score=np.nan,
        refit=True
    )

    fitted_gs = gs.fit(X, y, sample_weight=weights)
    print(fitted_gs.best_params_)
    if fitted_gs.best_params_['ratio_max_branch_length'] == 0:
        print(
            f'WARNING: Max distance set to 0, this means unweighted means performed best in gridsearch and '
            f'that NaNs/mean values will be predicted for all inputs (barring polytomies).')
    return fitted_gs


def bayes_opt(distance_matrix: pd.DataFrame, clf: bool, scorer, cv, X, y, weights=None, njobs=1, verbose=2):
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
    :return:
    '''
    from bayes_opt import BayesianOptimization

    def black_box_function(ratio, kappa):
        """Function with unknown internals we wish to maximize.

        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        phyln = PhylNearestNeighbours(distance_matrix, clf, ratio, kappa, fill_in_unknowns_with_mean=True)

        cv_score = cross_val_score(phyln, X, y, cv=cv, scoring=scorer, n_jobs=njobs, params={'sample_weight': weights})

        return np.mean(cv_score)

    # Bounded region of parameter space
    pbounds = {'ratio': (0, 1), 'kappa': (0, 3)}
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
        verbose=verbose
    )
    optimizer.maximize(init_points=10, n_iter=100
                       )

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

    best_phyln = PhylNearestNeighbours(distance_matrix, clf, ratio_max_branch_length=best_ratio, kappa=best_kappa, fill_in_unknowns_with_mean=True)
    best_phyln.fit(X, y, sample_weight=weights)
    return best_phyln
