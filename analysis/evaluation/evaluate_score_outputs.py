import os
import pathlib
from itertools import combinations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import brier_score_loss, mean_absolute_error

from analysis.imputation.helper_functions import get_input_data_paths, get_prediction_data_paths, missingness_types, nonstandard_sim_types, \
    number_of_simulation_iterations


def check_prediction_data(dfs: list[pd.DataFrame], ground_truth: pd.DataFrame, missing_values: pd.DataFrame):
    assert missing_values.shape[0] == ground_truth.shape[0]
    assert missing_values.shape[1] == ground_truth.shape[1]
    assert ground_truth.columns[0] == missing_values.columns[0]

    assert len(ground_truth.columns) == 1

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]

    for df in dfs:
        try:
            pd.testing.assert_index_equal(missing_values.index, df.index, check_names=False)

        except AssertionError:
            test_species = missing_values.index.tolist()
            pred_species = df.index.tolist()
            issues = [c for c in test_species if c not in pred_species]
            if issues != ['×_Staparesia_meintjesii', '×_Stapvalia_oskopensis']:
                # print(issues)
                # print('##############')
                # print([c for c in pred_species if c not in test_species])
                raise AssertionError(f'Model issue {df.columns[0]}')


def get_model_names(bin_or_cont):
    bin_model_names = ['corHMM', 'picante', 'phylnn_raw', 'phylnn_fill_means',
                       'logit_eigenvecs', 'logit_umap', 'logit_umap_supervised', 'logit_autoencoded', 'logit_autoenc_supervised',
                       'xgb_eigenvecs', 'xgb_umap', 'xgb_umap_supervised', 'xgb_autoencoded', 'xgb_autoenc_supervised']
    cont_model_names = ['phylopars', 'picante', 'phylnn_raw', 'phylnn_fill_means',
                        'linear_eigenvecs', 'linear_umap', 'linear_autoencoded',
                        'xgb_eigenvecs', 'xgb_umap', 'xgb_autoencoded']
    if bin_or_cont == 'binary':
        return bin_model_names
    elif bin_or_cont == 'continuous':
        return cont_model_names
    elif bin_or_cont == 'both':
        return list(set(bin_model_names + cont_model_names))
    else:
        raise ValueError(f'Unknown data type {bin_or_cont}')


def evaluate_output(real_or_sim: str, bin_or_cont: str, iteration: int, missing_type: str, drop_nans=False):
    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)
    out_dict = {}
    if real_or_sim == 'real_data':

        model_names = ['phylnn_raw', 'phylnn_fill_means']
    else:
        df_params = pd.read_csv(os.path.join(data_path, 'dataframe_params.csv'), index_col=0)
        try:
            lambda_ = df_params['lambda'].iloc[0]
            out_dict['lambda'] = lambda_
        except KeyError:
            lambda_ = None
        try:
            evmodel_ = df_params['model'].iloc[0]
        except KeyError:
            evmodel_ = real_or_sim

        try:
            kappa_ = df_params['kappa'].iloc[0]
            out_dict['kappa'] = kappa_
        except KeyError:
            kappa_ = None
        out_dict['Ev Model'] = evmodel_
        model_names = get_model_names(bin_or_cont)

    ground_truth = pd.read_csv(os.path.join(data_path, 'ground_truth.csv'), index_col=0)
    assert len(ground_truth.columns) == 1
    missing_values = pd.read_csv(os.path.join(data_path, f'{missing_type}_values.csv'), index_col=0)

    imputation_path = get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, missing_type)

    dfs = []
    for model_name in model_names:
        model_df = pd.read_csv(os.path.join(imputation_path, f'{model_name}.csv'), index_col=0)
        if bin_or_cont == 'binary':
            assert len(model_df.columns) == 2
            model_df = model_df[['1']]
        elif bin_or_cont == 'continuous':
            assert len(model_df.columns) == 1
        model_df.columns = [model_name]
        dfs.append(model_df)
    try:
        check_prediction_data(dfs, ground_truth, missing_values)
    except AssertionError as m:
        raise AssertionError(f'{m}. Issue with {real_or_sim}: str, {bin_or_cont}: str, {iteration}: int, {missing_type}. ')

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]
    test_ground_truths = ground_truth.loc[missing_values.index]

    full_df = test_ground_truths

    for df in dfs:
        full_df = pd.merge(full_df, df, left_index=True, right_index=True)

    issues = full_df[full_df['phylnn_fill_means'].isna()]
    assert len(issues) == 0

    if drop_nans:
        full_df = full_df[full_df['phylnn_raw'].notna()]
        pd.testing.assert_series_equal(full_df['phylnn_raw'], full_df['phylnn_fill_means'], check_names=False)
        full_df = full_df.drop('phylnn_fill_means', axis=1)
    else:
        full_df = full_df.drop('phylnn_raw', axis=1)
    if len(full_df) > 0:
        for model_name in model_names:
            if model_name in full_df.columns:
                if bin_or_cont == 'binary':
                    score = brier_score_loss(full_df[gt_target_name], full_df[model_name])
                elif bin_or_cont == 'continuous':
                    score = mean_absolute_error(full_df[gt_target_name], full_df[model_name])
                out_dict[model_name] = score
    return out_dict


def plot_results(df, model_names, out_dir, tag):
    plot_df = df[model_names]
    sns.boxplot(data=plot_df)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{tag}violin_plot.jpg'), dpi=300)
    plt.clf()
    plt.close()

    # if 'lambda' in df.columns:
    #
    #     for model_name in model_names:
    #         sns.jointplot(data=df, x='lambda', y=model_name, kind="reg")
    #         plt.xticks(rotation=30, ha='right')
    #         plt.ylabel(f'{model_name} Loss')
    #         plt.xlabel(f'Lambda')
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(out_dir, f'{tag}lambda_vs_{model_name}_loss.jpg'), dpi=300)
    #         plt.clf()
    #         plt.close()


def output_results_from_df(full_df: pd.DataFrame, out_dir: str, bin_or_cont: str, drop_nans: bool = False):
    if drop_nans:
        raise NotImplementedError('This is implemented but just clutters the results.')
    full_df = full_df.reset_index(drop=True)
    pathlib.Path(out_dir).mkdir(exist_ok=True, parents=True)

    tag = ''
    if drop_nans:
        tag = 'raw_'

    full_df.to_csv(os.path.join(out_dir, f'{tag}results.csv'))
    full_df.describe(include='all').to_csv(os.path.join(out_dir, f'{tag}results_summary.csv'))

    model_names = get_model_names(bin_or_cont)

    ttest_dict = {}
    for pair in list(combinations(model_names, 2)):
        model_name1, model_name2 = pair
        if model_name1 in full_df.columns and model_name2 in full_df.columns:
            t_stat, p_value = ttest_rel(full_df[model_name1], full_df[model_name2], nan_policy='omit')
            ttest_dict[f'{model_name1}_{model_name2}'] = [t_stat, p_value]
    # Convert to a DataFrame
    ttest_df = pd.DataFrame(ttest_dict, index=['stat', 'p value'])

    # Save to CSV
    ttest_df.to_csv(os.path.join(out_dir, f'{tag}ttest_results.csv'))

    plot_results(full_df, [c for c in model_names if c in full_df.columns], out_dir, tag)


def collate_simulation_outputs(real_or_sim: str, bin_or_cont: str, missing_type: str, drop_nans=False):
    if drop_nans:
        raise NotImplementedError('This is implemented but just clutters the results.')
    full_df = pd.DataFrame()
    for tag in range(1, number_of_simulation_iterations + 1):
        run_dict = evaluate_output(real_or_sim, bin_or_cont, tag, missing_type, drop_nans)
        run_df = pd.DataFrame(run_dict, index=[tag])
        full_df = pd.concat([full_df, run_df])
    out_dir = os.path.join('outputs', real_or_sim, bin_or_cont, missing_type)
    full_df['EV Model'] = real_or_sim
    full_df['Missing Type'] = missing_type
    output_results_from_df(full_df, out_dir, bin_or_cont, drop_nans)

    return full_df


def evaluate_all_combinations():
    binary_df = pd.DataFrame()

    cont_df = pd.DataFrame()

    for m in missingness_types:
        print(m)
        ### Standard
        bin_standard_df = collate_simulation_outputs('simulations', 'binary', m)

        binary_df = pd.concat([binary_df, bin_standard_df])

        cont_standard_df = collate_simulation_outputs('simulations', 'continuous', m)

        cont_df = pd.concat([cont_df, cont_standard_df])

        ### Extincts
        bin_extinct_df = collate_simulation_outputs('Extinct_BMT', 'binary', m)

        binary_df = pd.concat([binary_df, bin_extinct_df])

        cont_extinct_df = collate_simulation_outputs('Extinct_BMT', 'continuous', m)

        cont_df = pd.concat([cont_df, cont_extinct_df])

        # Nonstandard Simulations
        for sim_type in nonstandard_sim_types:
            bin_or_cont = nonstandard_sim_types[sim_type]
            sim_type_df = collate_simulation_outputs(sim_type, bin_or_cont, m)

            if bin_or_cont == 'binary':
                binary_df = pd.concat([binary_df, sim_type_df])

            if bin_or_cont == 'continuous':
                cont_df = pd.concat([cont_df, sim_type_df])

    output_results_from_df(binary_df, os.path.join('outputs', 'binary'), 'binary')

    output_results_from_df(cont_df, os.path.join('outputs', 'continuous'), 'continuous')


if __name__ == '__main__':
    evaluate_all_combinations()
