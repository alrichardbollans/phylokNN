import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from analysis.data.umapping import reduction_factor
from analysis.imputation.helper_functions import number_of_simulation_iterations, input_data_path, missingness_types, get_input_data_paths, \
    get_prediction_data_paths


def fit_and_output(clf_instance, out_dir, model_name, full_df, encoding_vars, target_name):
    X_test = full_df[full_df[target_name].isna()]
    X_train = full_df[~full_df[target_name].isna()]

    values = X_train[target_name].unique().tolist()
    if len(values) == 1:
        out_df = X_test.copy()
        if values[0] == 1:
            out_df[0] = 0
            out_df[1] = 1
            out_df = out_df[[0, 1]]
        elif values[0] == 0:
            out_df[0] = 1
            out_df[1] = 0
            out_df = out_df[[0, 1]]
        else:
            raise ValueError(f'Unexpected unique values: {values}')
    else:
        clf_instance.fit(X_train[encoding_vars], X_train[target_name])

        # y_pred = clf_instance.predict(X_test)
        y_proba = clf_instance.predict_proba(X_test[encoding_vars])
        out_df = pd.DataFrame(y_proba, index=X_test.index, columns=[0, 1])
        out_df.index.name = 'accepted_species'

    out_df.to_csv(os.path.join(out_dir, f'{model_name}.csv'))


def add_y_to_data(X, real_or_sim, bin_or_cont, iteration, missingness):
    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)
    y = pd.read_csv(os.path.join(data_path, f'{missingness}_values.csv'), index_col=0)

    target_name = y.columns.to_list()[0]

    df = pd.merge(y, X, left_index=True, right_index=True) # do this way to preserve ordering of y
    assert len(df) == len(y)
    df[target_name] = df.pop(target_name) # then move column to end
    encoding_vars = X.columns.to_list()

    return df,encoding_vars, target_name


def get_umap_data(real_or_sim, bin_or_cont, iteration):
    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)

    X = pd.read_csv(os.path.join(data_path, 'umap_unsupervised_embedding.csv'), index_col=0)

    return X


def get_eigenvectors(real_or_sim, bin_or_cont, iteration, reduction_fraction=reduction_factor):
    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)
    X = pd.read_csv(os.path.join(data_path, 'all_eigenvectors.csv'), index_col=0)
    num_cols_to_use = int(len(X.columns) * reduction_fraction)
    X = X.iloc[:, : num_cols_to_use]
    return X

def run_predictions():
    for iteration in tqdm(range(1, number_of_simulation_iterations + 1)):
        for m in missingness_types:
            for bin_or_cont in ['binary', 'continuous']:

                sim_list = ['simulations']
                if bin_or_cont == 'continuous':
                    sim_list += ['BMT', 'EB']
                if bin_or_cont == 'binary':
                    sim_list += ['BISSE', 'HISSE']

                for real_or_sim in sim_list:

                    umap_X = get_umap_data(real_or_sim, bin_or_cont, iteration)
                    umap_df, umap_encoding_vars, umap_target_name = add_y_to_data(umap_X, real_or_sim, bin_or_cont, iteration, m)

                    eigen_X = get_eigenvectors(real_or_sim, bin_or_cont, iteration)
                    eigen_df, eigen_encoding_vars, eigen_target_name = add_y_to_data(eigen_X, real_or_sim, bin_or_cont, iteration, m)

                    if bin_or_cont == 'binary':
                        out_dir = get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, m)
                        clf_instance = LogisticRegression()
                        fit_and_output(clf_instance, out_dir, 'logit_umap', umap_df, umap_encoding_vars, umap_target_name)

                        clf_instance = LogisticRegression()
                        fit_and_output(clf_instance, out_dir, 'logit_eigenvecs', eigen_df, eigen_encoding_vars, eigen_target_name)


if __name__ == '__main__':
    run_predictions()
