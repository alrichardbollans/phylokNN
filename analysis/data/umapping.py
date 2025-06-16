import os.path

import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from analysis.imputation.helper_functions import number_of_simulation_iterations, input_data_path

reduction_factor = 0.1


def umap_distances(distances: pd.DataFrame, reduction_fraction=reduction_factor):
    scaled_penguin_data = StandardScaler().fit_transform(distances)

    reducer = umap.UMAP(n_components=int(len(distances.columns) * reduction_fraction))

    embedding = reducer.fit_transform(scaled_penguin_data)
    umap_embedding = pd.DataFrame(embedding, index=distances.index)
    return umap_embedding


def unsupervised(dir_path: str, reduction_fraction=reduction_factor):
    distances = pd.read_csv(os.path.join(dir_path, 'tree_distances.csv'), index_col=0)

    umap_embedding = umap_distances(distances, reduction_fraction)
    umap_embedding.to_csv(os.path.join(dir_path, 'umap_unsupervised_embedding.csv'))


def main():
    for tag in tqdm(range(1, number_of_simulation_iterations + 1)):
        for var_type in ['binary', 'continuous']:
            dir_path = os.path.join(input_data_path, 'simulations', var_type, str(tag))
            unsupervised(dir_path)

            dir_path = os.path.join(input_data_path, 'real_data', var_type, str(tag))
            unsupervised(dir_path)

            if var_type == 'continuous':
                for ns in ['BMT', 'EB']:
                    dir_path = os.path.join(input_data_path, 'non_standard_simulations', ns, var_type, str(tag))

                    unsupervised(dir_path)
            if var_type == 'binary':
                for ns in ['BISSE', 'HISSE']:
                    dir_path = os.path.join(input_data_path, 'non_standard_simulations', ns, var_type, str(tag))

                    unsupervised(dir_path)
            dir_path = os.path.join(input_data_path, 'non_ultrametric_simulations', 'Extinct_BMT', var_type, str(tag))
            unsupervised(dir_path)


if __name__ == '__main__':
    main()
