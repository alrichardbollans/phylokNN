import os.path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from analysis.imputation.evaluate_binary_outputs import collate
from analysis.imputation.phylnn_predictions import continuous_output_path, continuous_input_path


def evaluate_continuous_output(tag):
    ground_truth = pd.read_csv(os.path.join(continuous_input_path, tag, 'simData_FinalData.csv'), index_col=0)
    missing_values = pd.read_csv(os.path.join(continuous_input_path, tag, 'mcar_values.csv'), index_col=0)
    my_predictions = pd.read_csv(os.path.join(continuous_output_path, tag, 'phylnn.csv'), index_col=0)[['0']]
    my_predictions.columns = ['phyloKNN']
    Rphylopars_predictions = pd.read_csv(os.path.join(continuous_output_path, tag, 'Rphylopars.csv'), index_col=0)
    Rphylopars_predictions.columns = ['Rphylopars']

    gt_target_name = ground_truth.columns[0]
    assert abs(ground_truth[gt_target_name].mean()) < 0.00001  # check standardisation
    assert abs(ground_truth[gt_target_name].std() - 1) < 0.00001  # check standardisation
    missing_values = missing_values[missing_values[gt_target_name].isna()]
    test_ground_truths = ground_truth.loc[missing_values.index]

    full_df = pd.merge(test_ground_truths, my_predictions, left_index=True, right_index=True)
    full_df = pd.merge(full_df, Rphylopars_predictions, left_index=True, right_index=True)

    nans_from_mine = full_df[full_df['phyloKNN'].isna()]
    if len(nans_from_mine) > 0:
        print(f'Warning: dropping {len(nans_from_mine)} nans from analysis.')
        full_df = full_df.dropna(subset=['phyloKNN'])

    phylopars_score = mean_absolute_error(full_df[gt_target_name], full_df['Rphylopars'])
    phylnn_score = mean_absolute_error(full_df[gt_target_name], full_df['phyloKNN'])
    return phylnn_score, phylopars_score


if __name__ == '__main__':
    collate(continuous_input_path, os.path.join('evaluation', 'continuous'), evaluate_continuous_output, 'Rphylopars')
