import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from analysis.imputation.helper_functions import number_of_simulation_iterations, input_data_path


def tsne_plot(df: pd.DataFrame, vars_to_encode: list, target_name: str, outfile, seaborn_kwargs=None):
    # T-SNE (t-Distributed Stochastic Neighbor Embedding) is a dataset decomposition technique which reduced the
    # dimensions of data and produces only top n components with maximum information.
    if seaborn_kwargs is None:
        seaborn_kwargs = {}
    tsne = TSNE(n_components=2, random_state=0)
    assert target_name not in vars_to_encode
    X_encoded = pd.DataFrame(tsne.fit_transform(df[vars_to_encode], ))
    X_encoded[target_name] = df[target_name].values
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=X_encoded, x=0, y=1, hue=target_name, **seaborn_kwargs)
    plt.savefig(outfile, dpi=300)
    plt.close()


def do_folder(dir_path: str):
    ground_truth = pd.read_csv(os.path.join(dir_path, 'ground_truth.csv'), index_col=0)
    target_name = ground_truth.columns.to_list()[0]

    eigen_vecs = pd.read_csv(os.path.join(dir_path, 'all_eigenvectors.csv'), index_col=0)
    umaps = pd.read_csv(os.path.join(dir_path, 'umap_unsupervised_embedding.csv'), index_col=0)

    full_df = pd.merge(ground_truth, umaps, left_index=True, right_index=True)

    tsne_plot(full_df, umaps.columns.tolist(), target_name, os.path.join(dir_path, 'umap_tsne_plot.png'))

    full_df = pd.merge(ground_truth, eigen_vecs, left_index=True, right_index=True)
    tsne_plot(full_df, eigen_vecs.columns.tolist(), target_name, os.path.join(dir_path, 'all_eigenvecs_tsne_plot.png'))

    broken_stick_params = pd.read_csv(os.path.join(dir_path, 'broken_stick_parameters.csv'), index_col=0)
    num_cols_to_use = broken_stick_params['broken_stick_number'].iloc[0]
    eigen_vecs = eigen_vecs.iloc[:, : num_cols_to_use]
    tsne_plot(full_df, eigen_vecs.columns.tolist(), target_name, os.path.join(dir_path, 'brokenstick_eigenvecs_tsne_plot.png'))


def main():
    for tag in tqdm(range(1, 2)):  # number_of_simulation_iterations + 1)):
        for var_type in ['binary', 'continuous']:
            dir_path = os.path.join(input_data_path, 'simulations', var_type, str(tag))

            do_folder(dir_path)

            if var_type == 'continuous':
                for ns in ['BMT', 'EB']:
                    dir_path = os.path.join(input_data_path, 'non_standard_simulations', ns, var_type, str(tag))

                    do_folder(dir_path)
            if var_type == 'binary':
                for ns in ['BISSE', 'HISSE']:
                    dir_path = os.path.join(input_data_path, 'non_standard_simulations', ns, var_type, str(tag))

                    do_folder(dir_path)


if __name__ == '__main__':
    main()
