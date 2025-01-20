import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import brier_score_loss
import seaborn as sns

from analysis.imputation.phylnn_predictions import binary_input_path, binary_output_path


def evaluate_bin_output(tag):
    ground_truth = pd.read_csv(os.path.join(binary_input_path, tag, 'simData_FinalData.csv'), index_col=0)
    missing_values = pd.read_csv(os.path.join(binary_input_path, tag, 'mcar_values.csv'), index_col=0)
    my_predictions = pd.read_csv(os.path.join(binary_output_path, tag, 'phylnn.csv'), index_col=0)[['1']]
    my_predictions.columns = ['PhyloNN']
    RcorHMM_predictions = pd.read_csv(os.path.join(binary_output_path, tag, 'corHMM.csv'), index_col=0)[['V2']]
    RcorHMM_predictions.columns = ['corHMM']

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]
    test_ground_truths = ground_truth.loc[missing_values.index]

    full_df = pd.merge(test_ground_truths, my_predictions, left_index=True, right_index=True)
    full_df = pd.merge(full_df, RcorHMM_predictions, left_index=True, right_index=True)

    nans_from_mine = full_df[full_df['PhyloNN'].isna()]
    if len(nans_from_mine) > 0:
        print(f'Warning: dropping {len(nans_from_mine)} nans from analysis.')
        full_df = full_df.dropna(subset=['PhyloNN'])

    if len(full_df) == 0:
        print(f'Warning: no data left for tag:{tag}')
        return None, None
    else:
        corHMM_score = brier_score_loss(full_df[gt_target_name], full_df['corHMM'])
        phylnn_score = brier_score_loss(full_df[gt_target_name], full_df['PhyloNN'])
        return phylnn_score, corHMM_score


def collate():
    phylnn_scores = []
    corHMM_scores = []

    phylnn_better = []
    corHMM_better = []
    for tag in range(1, 11):
        phylnn_score, corHMM_score = evaluate_bin_output(str(tag))
        if phylnn_score is not None:
            phylnn_scores.append(phylnn_score)
            corHMM_scores.append(corHMM_score)
            param_df = pd.read_csv(os.path.join(binary_input_path, str(tag), 'dataframe_params.csv'), index_col=0)
            if phylnn_score < corHMM_score:
                phylnn_better.append(param_df)
            elif corHMM_score < phylnn_score:
                corHMM_better.append(param_df)
    phylnn_better_df = pd.concat(phylnn_better)
    if len(corHMM_better) == 0:
        corHMM_better_df = pd.DataFrame()
    else:
        corHMM_better_df = pd.concat(corHMM_better)
        corHMM_better_df.describe(include='all').to_csv(os.path.join('evaluation', 'binary', 'corHMM_better_params_stats.csv'))


    phylnn_better_df.to_csv(os.path.join('evaluation', 'binary', 'phylnn_better_params.csv'))
    phylnn_better_df.describe(include='all').to_csv(os.path.join('evaluation', 'binary', 'phylnn_better_params_stats.csv'))
    corHMM_better_df.to_csv(os.path.join('evaluation', 'binary', 'corHMM_better_params.csv'))

    t_stat, p_value = ttest_rel(phylnn_scores, corHMM_scores)
    # Prepare the data for CSV
    results = {
        "Test": ["Paired t-test"],
        "t-statistic": [t_stat],
        "p-value": [p_value],
        "Phylnn Mean": [np.mean(phylnn_scores)],
        "corHMM Mean": [np.mean(corHMM_scores)],
        "Phylnn better": [len(phylnn_better_df)],
        "corHMM better": [len(corHMM_better_df)],
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(os.path.join('evaluation', 'binary', "ttest_results.csv"), index=False)

    plot_df = pd.DataFrame({'PhyloNN': phylnn_scores, 'Rphylopars': corHMM_scores})
    sns.violinplot(data=plot_df, fill=False)
    plt.savefig(os.path.join('evaluation', 'binary', 'violin_plot.jpg'), dpi=300)

if __name__ == '__main__':
    collate()