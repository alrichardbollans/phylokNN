import os.path
import pathlib

import numpy as np
import pandas as pd
from wcvpy.wcvp_download import get_all_taxa, wcvp_accepted_columns, wcvp_columns
from wcvpy.wcvp_name_matching import get_accepted_wcvp_info_from_ipni_ids_in_column

from analysis.data.real_data.continuous.get_genus_diversity_metrics import FAMILIES_OF_INTEREST

WCVP_VERSION = '12'


def prepare_MPNS_data() -> pd.DataFrame:
    mpns_df = pd.read_csv('mpns_v12_plants.csv', sep='|')
    mpns_df = mpns_df.drop(columns=['family'])
    all_taxa = get_all_taxa(version=WCVP_VERSION)
    accepted_mpns_df = get_accepted_wcvp_info_from_ipni_ids_in_column(mpns_df, 'ipni_id', all_taxa)

    gentianales_data = all_taxa[all_taxa[wcvp_columns['status']] == 'Accepted']
    gentianales_data = gentianales_data[gentianales_data[wcvp_accepted_columns['family']].isin(FAMILIES_OF_INTEREST)]
    gentianales_data = gentianales_data[gentianales_data[wcvp_accepted_columns['rank']] == 'Species']

    gentianales_data['Medicinal'] = gentianales_data['accepted_species'].apply(
        lambda x: 1 if x in accepted_mpns_df['accepted_species'].values else 0)
    gentianales_data[['accepted_species', 'Medicinal']].to_csv('binary_gentianales.csv', index=False)
    return gentianales_data

def summarise():
    df = pd.read_csv('binary_gentianales.csv')
    df.describe(include='all').to_csv('mpns_gentianales_summary.csv')

def get_an_mcar_sample():
    df = pd.read_csv('binary_gentianales.csv')
    out_path = '1'
    pathlib.Path(out_path).mkdir(exist_ok=True)
    df.to_csv(os.path.join(out_path, 'ground_truth.csv'), index=False)

    total_data = len(df)
    print(total_data)
    df.loc[df.sample(frac=0.1).index, 'Medicinal'] = np.nan
    nan_data = df[df['Medicinal'].isna()]
    print(len(nan_data))
    ratio = len(nan_data) / total_data
    print(ratio)
    assert ratio >0.0999 and ratio < 0.1001

    df.to_csv(os.path.join(out_path, 'mcar_values.csv'), index=False)

if __name__ == '__main__':
    # prepare_MPNS_data()
    summarise()
    get_an_mcar_sample()
