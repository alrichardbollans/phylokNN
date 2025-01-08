import os
import pathlib
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.model_selection import KFold

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
        val_scorer = mean_absolute_error
        greater_is_better = False
    else:
        clf = True
        out_dir = os.path.join(binary_output_path, tag)
        val_scorer = brier_score_loss
        greater_is_better = False

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    distance_df = pd.read_csv(distance_csv, index_col=0)
    missing_values_df = pd.read_csv(missing_values_csv, index_col=0)
    assert len(distance_df) == len(missing_values_df)
    assert 1 == len(missing_values_df.columns)

    target_name = missing_values_df.columns[0]
    train = missing_values_df[~missing_values_df[target_name].isna()]
    hparams = get_gridsearch_best_hparams_for_phylnn(train, train[target_name],
                                                     distance_df,
                                                     clf=clf, cv=KFold(n_splits=5, shuffle=True, random_state=42), val_scorer=val_scorer,
                                                     greater_is_better=greater_is_better)
    pickle.dump(hparams, open(os.path.join(out_dir, 'phylnn_hparams.pkl'), 'wb'))
    base_example = PhylNearestNeighbours(distance_df, clf=clf, ratio_max_branch_length=hparams['ratio_max_branch_length'], kappa=hparams['kappa'])
    base_example.fit(train.index, train[target_name])

    test = missing_values_df[missing_values_df[target_name].isna()]
    assert len(set(train.index).intersection(set(test.index))) == 0
    prediction = base_example.predict_proba(test.index)

    pd.DataFrame(prediction, index=test.index).to_csv(os.path.join(out_dir, 'phylnn.csv'))


def main():
    for tag in range(1, 11):
        predict(os.path.join(continuous_input_path, str(tag), 'tree_distances.csv'), os.path.join(continuous_input_path, str(tag), 'mcar_values.csv'),
                str(tag), True)

        predict(os.path.join(binary_input_path, str(tag), 'tree_distances.csv'), os.path.join(binary_input_path, str(tag), 'mcar_values.csv'),
                str(tag), False)


if __name__ == '__main__':
    main()
