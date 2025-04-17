import os

import pandas as pd
from tqdm import tqdm

from analysis.data.umapping import reduction_factor
from analysis.imputation.helper_functions import number_of_simulation_iterations, input_data_path
from phyloAutoEncoder import autoencode_pairwise_distances


def main():
    for tag in tqdm(range(1, number_of_simulation_iterations + 1)):
        for var_type in ['binary', 'continuous']:
            dir_path = os.path.join(input_data_path, 'simulations', var_type, str(tag))

            distances = pd.read_csv(os.path.join(dir_path, 'tree_distances.csv'), index_col=0)
            unsup_model, unsupervised = autoencode_pairwise_distances(distances, reduction_factor, dir_path)
            unsupervised.to_csv(os.path.join(dir_path, 'unsupervised_autoencoded_phylogeny.csv'))

            if var_type == 'continuous':
                ns_list = ['BMT', 'EB']

            if var_type == 'binary':
                ns_list = ['BISSE', 'HISSE']
            for ns in ns_list:
                dir_path = os.path.join(input_data_path, 'non_standard_simulations', ns, var_type, str(tag))

                distances = pd.read_csv(os.path.join(dir_path, 'tree_distances.csv'), index_col=0)
                unsup_model, unsupervised = autoencode_pairwise_distances(distances, reduction_factor, dir_path)
                unsupervised.to_csv(os.path.join(dir_path, 'unsupervised_autoencoded_phylogeny.csv'))


if __name__ == '__main__':
    main()
