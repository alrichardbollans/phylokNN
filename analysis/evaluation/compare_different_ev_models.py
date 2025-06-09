import os
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.evaluation.evaluate_score_outputs import get_model_names
from analysis.imputation.helper_functions import missingness_types, nonstandard_sim_types, number_of_simulation_iterations

bin_model_names = get_model_names('binary')
bin_model_names.remove('phylnn_raw')
cont_model_names = get_model_names('continuous')
cont_model_names.remove('phylnn_raw')
rename_models_and_ev_models = {'phylnn_fill_means': 'phylokNN', 'logit_eigenvecs': 'Eigenvec (L)', 'logit_umap': 'UMAP (L)',
                               'logit_umap_supervised': 'UMAP* (L)', 'logit_autoencoded': 'Autoenc (L)', 'logit_autoenc_supervised': 'Autoenc* (L)',
                               'xgb_eigenvecs': 'Eigenvec (XGB)', 'xgb_umap': 'UMAP (XGB)',
                               'xgb_umap_supervised': 'UMAP* (XGB)', 'xgb_autoencoded': 'Autoenc (XGB)', 'xgb_autoenc_supervised': 'Autoenc* (XGB)',
                               'linear_eigenvecs': 'Eigenvec (L)', 'linear_umap': 'UMAP (L)',
                               'linear_umap_supervised': 'UMAP* (L)', 'linear_autoencoded': 'Autoenc (L)',
                               'linear_autoenc_supervised': 'Autoenc* (L)',
                               }

binary_model_order = ['corHMM', 'picante', 'phylokNN', 'Eigenvec (L)', 'Eigenvec (XGB)', 'UMAP (L)', 'UMAP* (L)', 'UMAP (XGB)', 'UMAP* (XGB)',
                      'Autoenc (L)', 'Autoenc* (L)', 'Autoenc (XGB)', 'Autoenc* (XGB)']

continuous_model_order = ['phylopars', 'picante', 'phylokNN', 'Eigenvec (L)', 'Eigenvec (XGB)', 'UMAP (L)', 'UMAP (XGB)',
                          'Autoenc (L)', 'Autoenc (XGB)']


def check_scales():
    # This is just a sanity check to check the scaling
    EB_values = []
    BMT_values = []
    standard_values = []

    for tag in range(1, number_of_simulation_iterations + 1):
        tag = str(tag)
        eb_val = pd.read_csv(os.path.join('..', 'data', 'non_standard_simulations', 'EB', 'continuous', tag, 'ground_truth.csv'))[
            'trait_EB_scaled'].tolist()
        EB_values += eb_val

        bmt_val = pd.read_csv(os.path.join('..', 'data', 'non_standard_simulations', 'BMT', 'continuous', tag, 'ground_truth.csv'))[
            'trait_BM_trend_scaled'].tolist()
        BMT_values += bmt_val

        standard_val = pd.read_csv(os.path.join('..', 'data', 'simulations', 'continuous', tag, 'ground_truth.csv'))[
            'F1.1/1'].tolist()
        standard_values += standard_val
    print('EB values:', np.mean(EB_values))
    print('BMT values:', np.mean(BMT_values))
    print('Standard values:', np.mean(standard_values))
    plot_df = pd.DataFrame({'standard_values': standard_values, 'EB_values': EB_values, 'BMT_values': BMT_values})
    sns.boxplot(plot_df)
    plt.show()

def output_df(df, bin_or_cont, out_dir, group:str = 'EV Model'):
    for ev_model in df[group].unique():
        ev_df = df[df[group] == ev_model]
        ev_df =ev_df.sort_values(by=['Mean Loss'])
        filename = "".join(x for x in ev_model if x.isalnum())
        out_path = os.path.join(out_dir,bin_or_cont)
        pathlib.Path(out_path).mkdir(exist_ok=True, parents=True)
        ev_df.to_csv(os.path.join(out_path,f'{filename}_mean_results.csv'))


def plot_binary_and_continuous_cases(bin_df, cont_df, out_dir):
    sns.set_theme()

    ## do a useful plot
    plot_df = bin_df.groupby('EV Model').mean(numeric_only=True)
    plot_df = plot_df.reset_index()
    p_df = pd.melt(plot_df, id_vars='EV Model', value_vars=bin_model_names, var_name='Model', value_name='Mean Loss')
    p_df['EV Model'] = p_df['EV Model'].map({'simulations': 'ARD/SYM/ER', 'Extinct_BMT': 'BMT †', 'real_data':'MPNS'}).fillna(p_df['EV Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])

    ev_order = ['ARD/SYM/ER', 'BISSE', 'HISSE', 'BMT †', 'MPNS']
    p_df = p_df.sort_values(by="EV Model", key=lambda column: column.map(lambda e: ev_order.index(e)))
    output_df(p_df, 'binary',out_dir)
    g = sns.barplot(p_df, x='Model', y='Mean Loss', hue='EV Model', order=binary_model_order)

    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'binary_means.jpg'), dpi=300)
    plt.close()

    ## do a useful plot
    plot_df = cont_df.groupby('EV Model').mean(numeric_only=True)
    plot_df = plot_df.reset_index()
    p_df = pd.melt(plot_df, id_vars='EV Model', value_vars=cont_model_names, var_name='Model', value_name='Mean Loss')
    p_df['EV Model'] = p_df['EV Model'].map({'simulations': 'BM/OU', 'Extinct_BMT': 'BMT †', 'real_data':'BIEN'}).fillna(p_df['EV Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])
    ev_order = ['BM/OU', 'BMT', 'EB', 'BMT †','BIEN']
    p_df = p_df.sort_values(by="EV Model", key=lambda column: column.map(lambda e: ev_order.index(e)))
    g = sns.barplot(p_df, x='Model', y='Mean Loss', hue='EV Model', order=continuous_model_order)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'continuous_means.jpg'), dpi=300)
    output_df(p_df, 'continuous',out_dir)



def main():
    bin_df = pd.read_csv(os.path.join('outputs', 'binary', 'results.csv'))[bin_model_names + ['EV Model', 'Missing Type']]
    cont_df = pd.read_csv(os.path.join('outputs', 'continuous', 'results.csv'))[cont_model_names + ['EV Model', 'Missing Type']]

    plot_binary_and_continuous_cases(bin_df, cont_df, 'outputs')


if __name__ == '__main__':
    # check_scales()
    main()
