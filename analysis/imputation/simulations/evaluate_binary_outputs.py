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
    my_predictions.columns = ['phyloKNN']
    RcorHMM_predictions = pd.read_csv(os.path.join(binary_output_path, tag, 'corHMM.csv'), index_col=0)[['V2']]
    RcorHMM_predictions.columns = ['corHMM']

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]
    test_ground_truths = ground_truth.loc[missing_values.index]

    full_df = pd.merge(test_ground_truths, my_predictions, left_index=True, right_index=True)
    full_df = pd.merge(full_df, RcorHMM_predictions, left_index=True, right_index=True)

    nans_from_mine = full_df[full_df['phyloKNN'].isna()]
    if len(nans_from_mine) > 0:
        print(f'Warning: dropping {len(nans_from_mine)} nans from analysis.')
        full_df = full_df.dropna(subset=['phyloKNN'])

    if len(full_df) == 0:
        print(f'Warning: no data left for tag:{tag}')
        return None, None
    else:
        corHMM_score = brier_score_loss(full_df[gt_target_name], full_df['corHMM'])
        phylnn_score = brier_score_loss(full_df[gt_target_name], full_df['phyloKNN'])
        return phylnn_score, corHMM_score


def collate(input_path, output_path, eval_caller:callable, other_model_name:str):
    phylnn_scores = []
    other_model_scores = []

    phylnn_better = []
    other_model_better = []
    score_diffs = []

    lambdas = []
    for tag in range(1, 11):
        phylnn_score, other_score = eval_caller(str(tag))
        if phylnn_score is not None:
            phylnn_scores.append(phylnn_score)
            other_model_scores.append(other_score)
            score_diff = phylnn_score - other_score
            score_diffs.append(score_diff)
            param_df = pd.read_csv(os.path.join(input_path, str(tag), 'dataframe_params.csv'), index_col=0)
            if phylnn_score < other_score:
                phylnn_better.append(param_df)
            elif other_score < phylnn_score:
                other_model_better.append(param_df)

            lamba = param_df['lambda'].iloc[0]
            kappa = param_df['kappa'].iloc[0]
            lambdas.append(lamba)
    phylnn_better_df = pd.concat(phylnn_better)
    if len(other_model_better) == 0:
        other_model_better_df = pd.DataFrame()
    else:
        other_model_better_df = pd.concat(other_model_better)
        other_model_better_df.describe(include='all').to_csv(os.path.join(output_path, f'{other_model_name}_better_params_stats.csv'))


    phylnn_better_df.to_csv(os.path.join(output_path, 'phylnn_better_params.csv'))
    phylnn_better_df.describe(include='all').to_csv(os.path.join(output_path, 'phylnn_better_params_stats.csv'))
    other_model_better_df.to_csv(os.path.join(output_path, f'{other_model_name}_better_params.csv'))

    t_stat, p_value = ttest_rel(phylnn_scores, other_model_scores)
    # Prepare the data for CSV
    results = {
        "Test": ["Paired t-test"],
        "t-statistic": [t_stat],
        "p-value": [p_value],
        "Phylnn Mean": [np.mean(phylnn_scores)],
        f'{other_model_name} Mean': [np.mean(other_model_scores)],
        "Phylnn better": [len(phylnn_better_df)],
        f'{other_model_name} better': [len(other_model_better_df)],
    }

    # Convert to a DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(os.path.join(output_path, "ttest_results.csv"), index=False)

    plot_df = pd.DataFrame({'phyloKNN': phylnn_scores, f'{other_model_name}': other_model_scores})
    sns.violinplot(data=plot_df, fill=False)
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_path, 'violin_plot.jpg'), dpi=300)

    sns.jointplot(x=score_diffs, y=lambdas, kind="reg")
    plt.ylabel('Lambda')
    plt.xlabel(f'PhyloNN Loss - {other_model_name} Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'joint_plot.jpg'), dpi=300)

if __name__ == '__main__':
    collate(binary_input_path, os.path.join('evaluation', 'binary'), evaluate_bin_output, 'corHMM')