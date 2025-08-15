import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.data.helper_functions import simulation_types

from analysis.evaluation.compile_prediction_scores import read_all_results
from analysis.evaluation.helper_functions import bin_model_names, rename_models_and_ev_models, output_df, binary_model_order, cont_model_names, \
    continuous_model_order


def plot_binary_and_continuous_cases(bin_df, cont_df, group, out_dir):
    output_df(bin_df, 'binary', out_dir, group=group)
    output_df(cont_df, 'continuous', out_dir, group=group)


    sns.set_theme()
    # Error bars default is 95% confidence interval
    err_kws = {"color": ".2", "linewidth": 1.5}
    ## do a useful plot
    ## Turn  Model columns into useful row values
    p_df = pd.melt(bin_df, id_vars=group, value_vars=bin_model_names, var_name='Model', value_name='Mean Loss')
    # p_df['Ev Model'] = p_df['Ev Model'].map(
    #     {'simulations': 'ARD/SYM/ER', 'Extinct_BMT': 'BMT †', 'real_data': 'MPNS'}).fillna(p_df['Ev Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])
    if group == 'Ev Model':
        ev_order = simulation_types['binary']
        p_df = p_df.sort_values(by="Ev Model", key=lambda column: column.map(lambda e: ev_order.index(e)))

    g = sns.barplot(p_df, x='Model', y='Mean Loss', hue=group, order=binary_model_order,
                    capsize=.4,
                    err_kws=err_kws)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'binary_means.jpg'), dpi=300)
    plt.close()

    if group == 'Ev Model':
        # For readability don't plot semisupervised versions here
        p_df = p_df.sort_values(by="Model", key=lambda column: column.map(lambda e: binary_model_order.index(e)))
        g = sns.barplot(p_df[~p_df['Model'].str.contains('*', regex=False)], x=group, y='Mean Loss', hue='Model',
                        order=ev_order, capsize=.4,
                        err_kws=err_kws, )
        g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        sns.move_legend(
            g, "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title='Model', frameon=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'binary_means2.jpg'), dpi=300)
        plt.close()

    ## do a useful plot
    p_df = pd.melt(cont_df, id_vars=group, value_vars=cont_model_names, var_name='Model', value_name='Mean Loss')
    # p_df['Ev Model'] = p_df['Ev Model'].map(
    #     {'simulations': 'BM/OU', 'Extinct_BMT': 'BMT †', 'real_data': 'BIEN'}).fillna(p_df['Ev Model'])
    p_df['Model'] = p_df['Model'].map(rename_models_and_ev_models).fillna(p_df['Model'])
    if group == 'Ev Model':
        ev_order = simulation_types['continuous']
        p_df = p_df.sort_values(by="Ev Model", key=lambda column: column.map(lambda e: ev_order.index(e)))
    g = sns.barplot(p_df, x='Model', y='Mean Loss', hue=group, order=continuous_model_order,
                    capsize=.4,
                    err_kws=err_kws,
                    )
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'continuous_means.jpg'), dpi=300)
    plt.close()

    if group == 'Ev Model':
        p_df = p_df.sort_values(by="Model", key=lambda column: column.map(lambda e: continuous_model_order.index(e)))
        g = sns.barplot(p_df, x='Ev Model', y='Mean Loss', hue='Model', order=ev_order,
                        capsize=.4,
                        err_kws=err_kws,
                        )
        g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        sns.move_legend(
            g, "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title='Model', frameon=False,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'continuous_means2.jpg'), dpi=300)
        plt.close()


def main():
    bin_df, cont_df = read_all_results()
    plot_binary_and_continuous_cases(bin_df, cont_df,'Ev Model', 'ev_model_outputs')
    plot_binary_and_continuous_cases(bin_df, cont_df, 'Tree Type', 'tree_type_outputs')
    plot_binary_and_continuous_cases(bin_df, cont_df, 'Missing Type', 'missingness_outputs')


if __name__ == '__main__':
    main()
