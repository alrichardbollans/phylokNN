import os
import pathlib
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import brier_score_loss, mean_absolute_error
import seaborn as sns

from analysis.imputation.helper_functions import get_input_data_paths, get_prediction_data_paths, missingness_types, number_of_simulation_iterations
from phyloKNN import nan_safe_metric_wrapper


def check_prediction_data(dfs: list[pd.DataFrame], ground_truth: pd.DataFrame, missing_values: pd.DataFrame):
    assert missing_values.shape[0] == ground_truth.shape[0]
    assert missing_values.shape[1] == ground_truth.shape[1]
    assert ground_truth.columns[0] == missing_values.columns[0]

    assert len(ground_truth.columns) == 1

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]

    test_species = missing_values.index.tolist()
    for df in dfs:
        assert test_species == df.index.tolist()

def get_model_names(bin_or_cont):
    if bin_or_cont == 'binary':
        model_names = ['corHMM', 'phylnn_raw', 'phylnn_fill_means']
    elif bin_or_cont == 'continuous':
        model_names = ['phylopars', 'phylnn_raw', 'phylnn_fill_means']
    else:
        raise ValueError(f'Unknown data type {bin_or_cont}')
    return model_names

def evaluate_output(real_or_sim: str, bin_or_cont: str, iteration: int, missing_type: str):
    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)
    df_params = pd.read_csv(os.path.join(data_path, 'dataframe_params.csv'), index_col=0)
    lambda_ = df_params['lambda'].iloc[0]
    evmodel_ = df_params['model'].iloc[0]
    kappa_ = df_params['kappa'].iloc[0]
    ground_truth = pd.read_csv(os.path.join(data_path, 'ground_truth.csv'), index_col=0)
    assert len(ground_truth.columns) == 1
    missing_values = pd.read_csv(os.path.join(data_path, f'{missing_type}_values.csv'), index_col=0)

    imputation_path = get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, missing_type)

    model_names = get_model_names(bin_or_cont)

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

    check_prediction_data(dfs, ground_truth, missing_values)

    gt_target_name = ground_truth.columns[0]
    missing_values = missing_values[missing_values[gt_target_name].isna()]
    test_ground_truths = ground_truth.loc[missing_values.index]

    full_df = test_ground_truths

    for df in dfs:
        full_df = pd.merge(full_df, df, left_index=True, right_index=True)

    out_dict = {'lambda': lambda_, 'kappa': kappa_, 'Ev Model': evmodel_}
    for model_name in model_names:

        if bin_or_cont == 'binary':
            score = nan_safe_metric_wrapper(brier_score_loss)(full_df[gt_target_name], full_df[model_name])
        elif bin_or_cont == 'continuous':
            score = nan_safe_metric_wrapper(mean_absolute_error)(full_df[gt_target_name], full_df[model_name])
        out_dict[model_name] = score
    return out_dict


def collate(bin_or_cont: str, missing_type: str):
    full_df = pd.DataFrame()
    for tag in range(1, 9):#number_of_simulation_iterations + 1):
        run_dict = evaluate_output('simulations', bin_or_cont, tag, missing_type)
        run_df = pd.DataFrame(run_dict, index=[tag])
        full_df = pd.concat([full_df, run_df])
    out_dir = os.path.join('outputs','simulations', bin_or_cont, missing_type)
    pathlib.Path(out_dir).mkdir(exist_ok=True, parents=True)

    full_df.to_csv(os.path.join(out_dir,'results.csv'))
    full_df.describe(include='all').to_csv(os.path.join(out_dir,'results_summary.csv'))

    model_names = get_model_names(bin_or_cont)

    ttest_dict = {}
    for model_name in model_names:
        for model_name2 in model_names:

            t_stat, p_value = ttest_rel(full_df[model_name], full_df[model_name2])
            ttest_dict[f'{model_name}_{model_name2}'] = [t_stat,p_value]
    # Convert to a DataFrame
    ttest_df = pd.DataFrame(ttest_dict,index=['stat', 'p value'])

    # Save to CSV
    ttest_df.to_csv(os.path.join(out_dir, "ttest_results.csv"))

    plot_df = full_df[model_names]
    sns.violinplot(data=plot_df, fill=False)
    plt.ylabel('Loss')
    plt.savefig(os.path.join(out_dir, 'violin_plot.jpg'), dpi=300)
    plt.clf()
    plt.close()

    for model_name in model_names:
        sns.jointplot(data=full_df, x='lambda', y=model_name,kind="reg")
        plt.ylabel(f'{model_name} Loss')
        plt.xlabel(f'Lambda')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'lambda_vs_{model_name}_loss.jpg'), dpi=300)
        plt.clf()
        plt.close()
    # raise NotImplementedError('Need to change number of iteratinos')


def main():
    # Simulations
    for m in missingness_types:
        collate('binary', m)
        collate('continuous', m)


if __name__ == '__main__':
    main()
