import os.path
import pathlib
import pickle

import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, brier_score_loss
from sklearn.model_selection import KFold

from phyloKNN import nan_safe_metric_wrapper, phyloNN_bayes_opt, PhylNearestNeighbours

repo_path = os.environ.get('KEWSCRATCHPATH')
input_data_path = os.path.join(repo_path, 'phyloKNN', 'analysis', 'data')
prediction_path = os.path.join(repo_path, 'phyloKNN', 'analysis', 'imputation')

number_of_simulation_iterations = 100
missingness_types = ['mcar', 'phyloNa']


def get_iteration_bath_from_base(base: str, real_or_sim: str, bin_or_cont: str, iteration: int):
    if real_or_sim == 'real_data' or real_or_sim == 'simulations':
        basepath = os.path.join(base, real_or_sim)
    else:
        raise ValueError('Unknown real or simulation data')

    if bin_or_cont == 'binary' or bin_or_cont == 'continuous':
        nextpath = os.path.join(basepath, bin_or_cont)
    else:
        raise ValueError(f'Unknown data type {bin_or_cont}')

    iterpath = os.path.join(nextpath, str(iteration))

    return iterpath


def get_input_data_paths(real_or_sim: str, bin_or_cont: str, iteration: int):
    return get_iteration_bath_from_base(input_data_path, real_or_sim, bin_or_cont, iteration)


def get_prediction_data_paths(real_or_sim: str, bin_or_cont: str, iteration: int, missingness_type: str):
    return os.path.join(get_iteration_bath_from_base(prediction_path, real_or_sim, bin_or_cont, iteration), missingness_type)


def check_data(ground_truth, missing_values):
    assert len(ground_truth) == len(missing_values)
    assert len(ground_truth.columns) == len(missing_values.columns)

    def check_df(df):
        index_name = df.columns[0]
        assert index_name == 'accepted_species'

        assert len(df.columns) == 2

    check_df(ground_truth)
    check_df(missing_values)

    target_name = missing_values.columns[1]

    mcar_nans = missing_values[missing_values[target_name].isna()]
    assert len(mcar_nans) > 1
    assert len(mcar_nans) < len(missing_values)


def phylnn_predict(real_or_sim: str, bin_or_cont: str, iteration: int, missing_type: str):
    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)
    ground_truth = pd.read_csv(os.path.join(data_path, 'ground_truth.csv'))
    missing_values = pd.read_csv(os.path.join(data_path, f'{missing_type}_values.csv'))
    check_data(ground_truth, missing_values)
    if real_or_sim == 'real_data':
        assert iteration == 1
        if bin_or_cont == 'binary':
            distance_csv = os.path.join(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs', 'Species', 'species_distances.csv')
        elif bin_or_cont == 'continuous':
            distance_csv = os.path.join(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs', 'Genus', 'genus_distances.csv')
        else:
            raise ValueError(f'Unknown data type {bin_or_cont}')
    else:
        distance_csv = os.path.join(data_path, 'tree_distances.csv')

    if bin_or_cont == 'continuous':
        clf = False
        val_scorer = make_scorer(nan_safe_metric_wrapper(mean_absolute_error), greater_is_better=False)


    elif bin_or_cont == 'binary':
        clf = True
        val_scorer = make_scorer(nan_safe_metric_wrapper(brier_score_loss), greater_is_better=False, response_method='predict_proba')
    else:
        raise ValueError(f'Unknown data type {bin_or_cont}')

    out_dir = get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, missing_type)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    distance_df = pd.read_csv(distance_csv, index_col=0)
    # reduce size of distance dataframe to match trait data
    cols_for_distance_df = [c for c in distance_df.columns if c in missing_values[missing_values.columns[0]].values]
    distance_df = distance_df[cols_for_distance_df]
    distance_df = distance_df.loc[cols_for_distance_df]

    njobs = -1
    verbose = 0
    if clf and real_or_sim == 'real_data':
        # Some species are in the MPNS data but aren't in the tree, these seem to only be hybrids.
        unknown_cases = [c for c in missing_values['accepted_species'].values if c not in cols_for_distance_df]
        assert len(unknown_cases) < 200
        missing_values = missing_values[missing_values['accepted_species'].isin(cols_for_distance_df)]

        # in this case reduce computation load
        njobs = 1
        verbose = 1

    target_name = missing_values.columns[1]
    train = missing_values[~missing_values[target_name].isna()]

    best_ratio, best_kappa = phyloNN_bayes_opt(
        distance_df,
        clf=clf,
        scorer=val_scorer, cv=KFold(n_splits=5, shuffle=True, random_state=42), X=train, y=train[target_name].values, njobs=njobs, verbose=verbose)

    best_phyln_fill_means = PhylNearestNeighbours(distance_df, clf, ratio_max_branch_length=best_ratio, kappa=best_kappa,
                                                  fill_in_unknowns_with_mean=True)
    best_phyln_fill_means.fit(train, train[target_name].values)

    pickle.dump(best_phyln_fill_means, open(os.path.join(out_dir, 'phylnn_fill_means_hparams.pkl'), 'wb'))

    test = missing_values[missing_values[target_name].isna()]
    assert len(set(train.index).intersection(set(test.index))) == 0
    if clf:
        prediction_fill_means = best_phyln_fill_means.predict_proba(test)
    else:
        prediction_fill_means = best_phyln_fill_means.predict(test)

    pd.DataFrame(prediction_fill_means, index=test[test.columns[0]]).to_csv(os.path.join(out_dir, 'phylnn_fill_means.csv'))

    ## Without filling means
    best_phyln_raw = PhylNearestNeighbours(distance_df, clf, ratio_max_branch_length=best_ratio, kappa=best_kappa,
                                           fill_in_unknowns_with_mean=False)
    best_phyln_raw.fit(train, train[target_name].values)

    pickle.dump(best_phyln_raw, open(os.path.join(out_dir, 'phylnn_raw_hparams.pkl'), 'wb'))

    if clf:
        prediction_raw = best_phyln_raw.predict_proba(test)
    else:
        prediction_raw = best_phyln_raw.predict(test)

    pd.DataFrame(prediction_raw, index=test[test.columns[0]]).to_csv(os.path.join(out_dir, 'phylnn_raw.csv'))
