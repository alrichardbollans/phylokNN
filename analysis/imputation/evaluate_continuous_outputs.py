import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error
import seaborn as sns

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


def collate():
    phylnn_scores = []
    phylopars_scores = []

    phylnn_better = []
    phylopars_better = []
    for tag in range(1, 11):
        phylnn_score, phylopars_score = evaluate_continuous_output(str(tag))
        phylnn_scores.append(phylnn_score)
        phylopars_scores.append(phylopars_score)
        param_df = pd.read_csv(os.path.join(continuous_input_path, str(tag), 'dataframe_params.csv'), index_col=0)
        if phylnn_score < phylopars_score:
            phylnn_better.append(param_df)
        elif phylopars_score < phylnn_score:
            phylopars_better.append(param_df)
    phylnn_better_df = pd.concat(phylnn_better)
    phylopars_better_df = pd.concat(phylopars_better)

    phylnn_better_df.to_csv(os.path.join('evaluation', 'continuous', 'phylnn_better_params.csv'))
    phylnn_better_df.describe(include='all').to_csv(os.path.join('evaluation', 'continuous', 'phylnn_better_params_stats.csv'))
    phylopars_better_df.to_csv(os.path.join('evaluation', 'continuous', 'phylopars_better_params.csv'))
    phylopars_better_df.describe(include='all').to_csv(os.path.join('evaluation', 'continuous', 'phylopars_better_params_stats.csv'))

    t_stat, p_value = ttest_rel(phylnn_scores, phylopars_scores)
    # Prepare the data for CSV
    results = {
        "Test": ["Paired t-test"],
        "t-statistic": [t_stat],
        "p-value": [p_value],
        "Phylnn Mean": [np.mean(phylnn_scores)],
        "Rphylopars Mean": [np.mean(phylopars_scores)],
        "Phylnn better": [len(phylnn_better_df)],
        "Rphylopars better": [len(phylopars_better_df)],
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(os.path.join('evaluation', 'continuous', "ttest_results.csv"), index=False)

    plot_df = pd.DataFrame({'phyloKNN': phylnn_scores, 'Rphylopars': phylopars_scores})
    sns.violinplot(data=plot_df,fill=False)
    plt.savefig(os.path.join('evaluation', 'continuous', 'violin_plot.jpg'), dpi=300)


if __name__ == '__main__':
    collate()
