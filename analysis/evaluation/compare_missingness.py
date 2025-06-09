import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.evaluation.compare_different_ev_models import bin_model_names, cont_model_names, rename_models_and_ev_models, output_df, \
    binary_model_order, continuous_model_order


def plot_binary_and_continuous_cases(bin_df, cont_df, out_dir):
    sns.set_theme()

    ## do a useful plot
    plot_df = bin_df.groupby('Missing Type').mean(numeric_only=True)
    plot_df = plot_df.reset_index()
    p_df = pd.melt(plot_df, id_vars='Missing Type', value_vars=bin_model_names, var_name='Model', value_name='Mean Loss')
    # p_df['EV Model'] = p_df['EV Model'].map({'simulations': 'ARD/SYM/ER', 'Extinct_BMT': 'BMT †', 'real_data':'MPNS'}).fillna(p_df['EV Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])

    # ev_order = ['ARD/SYM/ER', 'BISSE', 'HISSE', 'BMT †', 'MPNS']
    # p_df = p_df.sort_values(by="EV Model", key=lambda column: column.map(lambda e: ev_order.index(e)))
    output_df(p_df, 'binary',out_dir, group='Missing Type')
    g = sns.barplot(p_df, x='Model', y='Mean Loss', hue='Missing Type', order=binary_model_order)

    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'binary_means.jpg'), dpi=300)
    plt.close()

    ## do a useful plot
    plot_df = cont_df.groupby('Missing Type').mean(numeric_only=True)
    plot_df = plot_df.reset_index()
    p_df = pd.melt(plot_df, id_vars='Missing Type', value_vars=cont_model_names, var_name='Model', value_name='Mean Loss')
    # p_df['EV Model'] = p_df['EV Model'].map({'simulations': 'BM/OU', 'Extinct_BMT': 'BMT †', 'real_data':'BIEN'}).fillna(p_df['EV Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])
    # ev_order = ['BM/OU', 'BMT', 'EB', 'BMT †','BIEN']
    # p_df = p_df.sort_values(by="EV Model", key=lambda column: column.map(lambda e: ev_order.index(e)))
    g = sns.barplot(p_df, x='Model', y='Mean Loss', hue='Missing Type', order=continuous_model_order)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'continuous_means.jpg'), dpi=300)
    output_df(p_df, 'continuous',out_dir, group='Missing Type')



def main():
    bin1_df = pd.read_csv(os.path.join('outputs', 'mcar_binary', 'results.csv'))[bin_model_names + ['Missing Type']]
    bin2_df = pd.read_csv(os.path.join('outputs', 'phyloNa_binary', 'results.csv'))[bin_model_names + ['Missing Type']]
    bin_df = pd.concat([bin1_df, bin2_df])
    
    cont1_df = pd.read_csv(os.path.join('outputs', 'mcar_continuous', 'results.csv'))[cont_model_names + ['Missing Type']]
    cont2_df = pd.read_csv(os.path.join('outputs', 'phyloNa_continuous', 'results.csv'))[cont_model_names + ['Missing Type']]
    cont_df = pd.concat([cont1_df, cont2_df])
    
    
    plot_binary_and_continuous_cases(bin_df, cont_df, 'missingness_outputs')

if __name__ == '__main__':
    main()