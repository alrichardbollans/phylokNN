import os

import pandas as pd

from analysis.imputation.helper_functions import missingness_types, nonstandard_sim_types

_alpha = 0.05

def ttests(eval_dir:str):
    test_results = pd.read_csv(os.path.join(eval_dir, 'ttest_results.csv'), index_col=0)
    significant_cases = []
    for c in test_results.columns:
        # if test_results[c].loc['p value']<_alpha:
            significant_cases.append([c, test_results[c].loc['stat'],test_results[c].loc['p value']])

    raw_test_results = pd.read_csv(os.path.join(eval_dir, 'raw_ttest_results.csv'), index_col=0)
    for c in raw_test_results.columns:
        # if raw_test_results[c].loc['p value']<_alpha:
            significant_cases.append([c, raw_test_results[c].loc['stat'],raw_test_results[c].loc['p value']])

    return pd.DataFrame(significant_cases, columns=['models', 'stat', 'p value'])



def hochberg_correction(df: pd.DataFrame, p_value_col: str):
    # Hochberg Method
    # Yosef Hochberg, ‘A Sharper Bonferroni Procedure for Multiple Tests of Significance’, Biometrika 75, no. 4 (1988): 800–802, https://doi.org/10.1093/biomet/75.4.800.


    # The adjustement is the same, but the usage is slightly different.
    new_df = df.sort_values(by=p_value_col)
    n = len(new_df.index)
    new_df.reset_index(inplace=True, drop=True)

    new_df['hochberg_adjusted_p_value'] = new_df.apply(lambda x: x[p_value_col] * (n - x.name), axis=1)

    return new_df

def summarise_ttests():
    full_df = pd.DataFrame()
    for m in missingness_types:
        print(m)
        out_dir = os.path.join('outputs', 'simulations', 'binary', m)
        bin_cases = ttests(out_dir)
        bin_cases['variable'] = 'binary'
        out_dir = os.path.join('outputs', 'simulations', 'continuous', m)
        continuous_cases = ttests(out_dir)
        continuous_cases['variable'] = 'continuous'

        missing_type_df = pd.concat([bin_cases, continuous_cases], ignore_index=True)
        missing_type_df['case'] = 'Standard Simulations'


        for sim_type in nonstandard_sim_types:
            ns_out_dir = os.path.join('outputs', sim_type, nonstandard_sim_types[sim_type], m)
            ns_cases = ttests(ns_out_dir)
            ns_cases['variable'] = nonstandard_sim_types[sim_type]

            ns_cases['case'] = sim_type

            missing_type_df = pd.concat([missing_type_df, ns_cases], ignore_index=True)


        missing_type_df['missing type'] = m
        full_df = pd.concat([full_df, missing_type_df])


    full_df  = hochberg_correction(full_df, p_value_col='p value')
    full_df.to_csv(os.path.join('outputs', 'summarised_ttest_results.csv'))

if __name__ == '__main__':
    summarise_ttests()