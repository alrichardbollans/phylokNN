import os.path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from analysis.imputation.phylnn_predictions import continuous_output_path, continuous_input_path


def evaluate_continuous_outputs(tag):
    ground_truth = pd.read_csv(os.path.join(continuous_input_path, tag, 'simData_FinalData.csv'), index_col=0)
    missing_values = pd.read_csv(os.path.join(continuous_input_path, tag, 'mcar_values.csv'), index_col=0)
    my_predictions = pd.read_csv(os.path.join(continuous_output_path, tag, 'phylnn.csv'), index_col=0)[['1']]
    my_predictions.columns = ['PhyloNN']
    Rphylopars_predictions = pd.read_csv(os.path.join(continuous_output_path, tag, 'Rphylopars.csv'), index_col=0)
    Rphylopars_predictions.columns = ['Rphylopars']

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]
    test_ground_truths = ground_truth.loc[missing_values.index]

    full_df = pd.merge(test_ground_truths, my_predictions, left_index=True, right_index=True)
    full_df = pd.merge(full_df, Rphylopars_predictions, left_index=True, right_index=True)

    phylopars_score = mean_absolute_error(full_df[gt_target_name], full_df['Rphylopars'])
    phylnn_score = mean_absolute_error(full_df[gt_target_name], full_df['PhyloNN'])
    print(f'phylopars score: {phylopars_score}')
    print(f'phylnn score: {phylnn_score}')
    print(
        f'phylopars vs phylnn: phylopars score: {phylopars_score} vs phylnn score: {phylnn_score} = {phylopars_score - phylnn_score}'
    )
    pass


if __name__ == '__main__':
    evaluate_continuous_outputs('1')
