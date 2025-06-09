import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

from analysis.evaluation.compare_different_ev_models import bin_model_names, rename_models_and_ev_models, binary_model_order
from analysis.evaluation.evaluate_score_outputs import collate_simulation_outputs, output_results_from_df


def evaluate_all_combinations():

    binary_df = collate_simulation_outputs('my_apm_data', 'binary', 'mcar', range_to_eval=10, scorer=average_precision_score)
    output_results_from_df(binary_df, os.path.join('outputs', 'my_apm_data'), 'binary', scorer_label='AP')


def plot_binary_cases(bin_df, out_dir):
    sns.set_theme()

    ## do a useful plot
    plot_df = bin_df.groupby('EV Model').mean(numeric_only=True)
    plot_df = plot_df.reset_index()
    p_df = pd.melt(plot_df, id_vars='EV Model', value_vars=bin_model_names, var_name='Model', value_name='Mean AP')
    p_df['EV Model'] = p_df['EV Model'].map({'simulations': 'ARD/SYM/ER', 'Extinct_BMT': 'BMT †'}).fillna(p_df['EV Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])

    g = sns.barplot(p_df, x='Model', y='Mean AP', order=binary_model_order)

    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'binary_means.jpg'), dpi=300)
    plt.close()


if __name__ == '__main__':
    evaluate_all_combinations()
    bin_model_names.remove('logit_umap_supervised')
    bin_model_names.remove('xgb_umap_supervised')
    bin_model_names.remove('logit_autoenc_supervised')
    bin_model_names.remove('xgb_autoenc_supervised')
    bin_df = pd.read_csv(os.path.join('outputs', 'my_apm_data', 'results.csv'))[bin_model_names + ['EV Model', 'Missing Type']]

    plot_binary_cases(bin_df, os.path.join('outputs', 'my_apm_data'))
