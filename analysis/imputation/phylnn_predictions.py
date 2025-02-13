import os
import pathlib
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss, make_scorer
from sklearn.model_selection import KFold
from tqdm import tqdm

from phyloKNN import phyloNN_bayes_opt, PhylNearestNeighbours, nan_safe_metric_wrapper

repo_path = os.environ.get('KEWSCRATCHPATH')
data_path = os.path.join(repo_path, 'PhyloNN', 'analysis', 'data')
simulations_path = os.path.join(data_path, 'simulations')
continuous_input_path = os.path.join(simulations_path, 'continuous')
binary_input_path = os.path.join(simulations_path, 'binary')

continuous_output_path = os.path.join('predictions', 'continuous')
binary_output_path = os.path.join('predictions', 'binary')

missingness_types = ['mar', 'mcar', 'mcar', 'phyloNa']


def predict(tag: str, continuous: bool):
    if continuous:
        clf = False
        parent_out_dir = os.path.join(continuous_output_path, tag)
        val_scorer = make_scorer(nan_safe_metric_wrapper(mean_absolute_error), greater_is_better=False)
        input_path = os.path.join(continuous_input_path, str(tag))
    else:
        clf = True
        parent_out_dir = os.path.join(binary_output_path, tag)
        val_scorer = make_scorer(nan_safe_metric_wrapper(brier_score_loss), greater_is_better=False)
        input_path = os.path.join(binary_input_path, str(tag))

    distance_csv = os.path.join(input_path, 'tree_distances.csv')
    distance_df = pd.read_csv(distance_csv, index_col=0)

    for missingness in missingness_types:
        out_dir = os.path.join(parent_out_dir, missingness)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        missing_values_csv = os.path.join(input_path, f'{missingness}_values.csv')
        missing_values_df = pd.read_csv(missing_values_csv)
        assert len(distance_df) == len(missing_values_df)
        assert 2 == len(missing_values_df.columns)
        target_name = missing_values_df.columns[1]
        train = missing_values_df[~missing_values_df[target_name].isna()]

        best_ratio, best_kappa = phyloNN_bayes_opt(
            distance_df,
            clf=clf,
            scorer=val_scorer, cv=KFold(n_splits=5, shuffle=True, random_state=42), X=train, y=train[target_name].values, njobs=-1, verbose=0)

        best_phyln_fill_means = PhylNearestNeighbours(distance_df, clf, ratio_max_branch_length=best_ratio, kappa=best_kappa,
                                                      fill_in_unknowns_with_mean=True)
        best_phyln_fill_means.fit(train, train[target_name].values)

        pickle.dump(best_phyln_fill_means, open(os.path.join(out_dir, 'phylnn_fill_means_hparams.pkl'), 'wb'))

        test = missing_values_df[missing_values_df[target_name].isna()]
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


def main():
    for tag in tqdm(range(1, 1001)):
        predict(str(tag), True)

        predict(str(tag), False)


if __name__ == '__main__':
    main()
