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

    # raw_test_results = pd.read_csv(os.path.join(eval_dir, 'raw_ttest_results.csv'), index_col=0)
    # for c in raw_test_results.columns:
    #     # if raw_test_results[c].loc['p value']<_alpha:
    #         significant_cases.append([c, raw_test_results[c].loc['stat'],raw_test_results[c].loc['p value']])

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

    bin_cases = ttests(os.path.join('outputs', 'binary'))
    bin_cases['variable'] = 'binary'
    bin_df = hochberg_correction(bin_cases, p_value_col='p value')
    bin_df.to_csv(os.path.join('outputs', 'binary_ttest_summary.csv'))


    continuous_cases = ttests(os.path.join('outputs', 'continuous'))
    continuous_cases['variable'] = 'continuous'
    continuous_df = hochberg_correction(continuous_cases, p_value_col='p value')
    continuous_df.to_csv(os.path.join('outputs', 'continuous_ttest_summary.csv'))

if __name__ == '__main__':
    summarise_ttests()