import os

import pandas as pd

from analysis.data.umapping import reduction_factor, unsupervised
from analysis.imputation.helper_functions import input_data_path
from phyloAutoEncoder import autoencode_pairwise_distances


def main():
    dir_path = os.path.join(input_data_path, 'my_apm_data', 'binary', str(1))
    unsupervised(dir_path)

    distances = pd.read_csv(os.path.join(dir_path, 'tree_distances.csv'), index_col=0)
    unsup_model, unsupervised_df = autoencode_pairwise_distances(distances, reduction_factor, dir_path)
    unsupervised_df.to_csv(os.path.join(dir_path, 'unsupervised_autoencoded_phylogeny.csv'))

if __name__ == '__main__':
    main()