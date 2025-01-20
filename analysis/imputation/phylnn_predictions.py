import os
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss, make_scorer
from sklearn.model_selection import KFold, GridSearchCV

from phyloNN.phylogenetic_neighbours_model import PhylNearestNeighbours, get_gridsearch_best_hparams_for_phylnn

repo_path = os.environ.get('KEWSCRATCHPATH')
data_path = os.path.join(repo_path, 'PhyloNN', 'analysis', 'data')
simulations_path = os.path.join(data_path, 'simulations')
continuous_input_path = os.path.join(simulations_path, 'continuous')
binary_input_path = os.path.join(simulations_path, 'binary')

continuous_output_path = os.path.join('predictions', 'continuous')
binary_output_path = os.path.join('predictions', 'binary')


def predict(distance_csv: str, missing_values_csv: str, tag: str, continuous: bool):
    if continuous:
        clf = False
        out_dir = os.path.join(continuous_output_path, tag)
        val_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    else:
        clf = True
        out_dir = os.path.join(binary_output_path, tag)
        val_scorer = make_scorer(brier_score_loss, greater_is_better=False)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    distance_df = pd.read_csv(distance_csv, index_col=0)
    missing_values_df = pd.read_csv(missing_values_csv)
    assert len(distance_df) == len(missing_values_df)
    assert 2 == len(missing_values_df.columns)
    target_name = missing_values_df.columns[1]
    train = missing_values_df[~missing_values_df[target_name].isna()]

    phyln = PhylNearestNeighbours(distance_df, clf=clf, ratio_max_branch_length=1,  kappa=1, fill_in_unknowns_with_mean=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=phyln,
        param_grid={'ratio_max_branch_length': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    'kappa': [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]},
        cv=cv,
        n_jobs=-1,
        scoring=val_scorer,
        verbose=1,
        error_score=np.nan,
        refit=False
    )

    fitted_gs = gs.fit(train, train[target_name].values)
    print(fitted_gs.best_params_)
    pickle.dump(fitted_gs, open(os.path.join(out_dir, 'phylnn_hparams.pkl'), 'wb'))

    best_phyln = PhylNearestNeighbours(distance_df, clf=clf, ratio_max_branch_length=fitted_gs.best_params_['ratio_max_branch_length'],
                                       kappa=fitted_gs.best_params_['kappa'], fill_in_unknowns_with_mean=False)

    best_phyln.fit(train, train[target_name].values)

    test = missing_values_df[missing_values_df[target_name].isna()]
    assert len(set(train.index).intersection(set(test.index))) == 0
    if clf:
        prediction = best_phyln.predict_proba(test)
    else:
        prediction = best_phyln.predict(test)

    pd.DataFrame(prediction, index=test[test.columns[0]]).to_csv(os.path.join(out_dir, 'phylnn.csv'))


def main():
    for tag in range(1, 11):
        predict(os.path.join(continuous_input_path, str(tag), 'tree_distances.csv'), os.path.join(continuous_input_path, str(tag), 'mcar_values.csv'),
                str(tag), True)

        predict(os.path.join(binary_input_path, str(tag), 'tree_distances.csv'), os.path.join(binary_input_path, str(tag), 'mcar_values.csv'),
                str(tag), False)


if __name__ == '__main__':
    main()
