import os

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from analysis.evaluation.evaluate_score_outputs import collate_simulation_outputs, output_results_from_df, get_model_names
from analysis.imputation.helper_functions import missingness_types, nonstandard_sim_types


def evaluate_standard_vs_nonstandard():
    bin_standard_df = pd.DataFrame()
    cont_standard_df = pd.DataFrame()

    bin_non_standard_df = pd.DataFrame()
    cont_non_standard_df = pd.DataFrame()

    bin_model_names = get_model_names('binary')
    bin_model_names.remove('phylnn_raw')
    cont_model_names = get_model_names('continuous')
    cont_model_names.remove('phylnn_raw')

    for m in missingness_types:
        print(m)
        bin_df = pd.read_csv(os.path.join('outputs', 'simulations', 'binary', m, 'results.csv'))[bin_model_names]

        bin_standard_df = pd.concat([bin_standard_df, bin_df])

        cont_df = pd.read_csv(os.path.join('outputs', 'simulations', 'continuous', m, 'results.csv'))[cont_model_names]
        cont_standard_df = pd.concat([cont_standard_df, cont_df])

        # Nonstandard Simulations
        for sim_type in nonstandard_sim_types:
            bin_or_cont = nonstandard_sim_types[sim_type]
            m_false_df = pd.read_csv(os.path.join('outputs', sim_type, bin_or_cont, m, 'results.csv'))
            if bin_or_cont == 'binary':
                bin_non_standard_df = pd.concat([bin_non_standard_df, m_false_df[bin_model_names]])

            elif bin_or_cont == 'continuous':
                cont_non_standard_df = pd.concat([cont_non_standard_df, m_false_df[cont_model_names]])
            else:
                raise ValueError('Invalid value for bin_or_cont')

    # output_results_from_df(bin_standard_df, os.path.join('outputs', 'binary standard evolution models'), 'binary', drop_nans=False)
    # output_results_from_df(cont_standard_df, os.path.join('outputs', 'continuous standard evolution models'), 'continuous', drop_nans=False)
    #
    # output_results_from_df(bin_non_standard_df, os.path.join('outputs', 'binary nonstandard evolution models'), 'binary', drop_nans=False)
    # output_results_from_df(cont_non_standard_df, os.path.join('outputs', 'continuous nonstandard evolution models'), 'continuous', drop_nans=False)

    ## do a useful plot
    plot_df = pd.DataFrame(bin_standard_df.mean())
    plot_df.columns = ['Mean Loss']
    plot_df['Model'] = plot_df.index
    plot_df['Type']='Standard'

    ns_plot_df = pd.DataFrame(bin_non_standard_df.mean())
    ns_plot_df.columns = ['Mean Loss']
    ns_plot_df['Model'] = ns_plot_df.index
    ns_plot_df['Type']='Non Standard'

    import seaborn as sns
    sns.barplot(pd.concat([plot_df, ns_plot_df]), x='Model', y='Mean Loss', hue='Type')
    plt.savefig(os.path.join('outputs', 'binary_means.jpg'))
    plt.close()

    ## do a useful plot
    plot_df = pd.DataFrame(cont_standard_df.mean())
    plot_df.columns = ['Mean Loss']
    plot_df['Model'] = plot_df.index
    plot_df['Type'] = 'Standard'

    ns_plot_df = pd.DataFrame(cont_non_standard_df.mean())
    ns_plot_df.columns = ['Mean Loss']
    ns_plot_df['Model'] = ns_plot_df.index
    ns_plot_df['Type'] = 'Non Standard'

    import seaborn as sns
    sns.barplot(pd.concat([plot_df, ns_plot_df]), x='Model', y='Mean Loss', hue='Type')
    plt.savefig(os.path.join('outputs', 'continuous_means.jpg'))



    ## Output ttest results
    ttest_dict = {}
    all_model_names = get_model_names('both')
    for model in all_model_names:
        if model in bin_standard_df.columns:
            bin_standard_results = bin_standard_df[model].dropna().values
            bin_nonstandard_results = bin_non_standard_df[model].dropna().values
            t_stat, p_value = ttest_ind(bin_standard_results, bin_nonstandard_results)
            ttest_dict[f'{model}_binary'] = [t_stat, p_value]
        if model in cont_standard_df.columns:
            cont_standard_results = cont_standard_df[model].dropna().values
            cont_nonstandard_results = cont_non_standard_df[model].dropna().values
            t_stat, p_value = ttest_ind(cont_standard_results, cont_nonstandard_results)
            ttest_dict[f'{model}_continuous'] = [t_stat, p_value]

    # Convert to a DataFrame
    ttest_df = pd.DataFrame(ttest_dict, index=['stat', 'p value'])

    # Save to CSV
    ttest_df.to_csv(os.path.join('outputs', f'standard_vs_nonstandard_ttest_results.csv'))


if __name__ == '__main__':
    evaluate_standard_vs_nonstandard()
