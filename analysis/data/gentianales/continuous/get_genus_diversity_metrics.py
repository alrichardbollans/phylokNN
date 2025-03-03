import os

import numpy as np
import pandas as pd
from phytochempy.chemical_diversity_metrics import calculate_FAD_measures
from wcvpy.wcvp_download import wcvp_accepted_columns

FAMILIES_OF_INTEREST = ['Gelsemiaceae', 'Gentianaceae', 'Apocynaceae', 'Loganiaceae', 'Rubiaceae']
COMPOUND_ID_COL = 'Standard_SMILES'
species_in_study_csv = os.path.join('inputs', 'species_in_study.csv')
all_species_compound_csv = os.path.join('inputs', 'all_species_compound_data.csv')  # From
# Richard-Bollans, Adam. “PhytoChemicalDiversity: 1.0.”, DOI:10.5281/zenodo.14778820, 2025.

div_metric = 'APWD'


def resolve_traits_to_group(df: pd.DataFrame, tag: str):
    # resolve traits to group
    assert len(df[df.duplicated(subset=['accepted_species', 'Assigned_group'])].index) == 0

    df['number_of_species_in_group'] = df[['Assigned_group', 'accepted_species']].groupby('Assigned_group').transform('count')
    df.to_csv(os.path.join('outputs', 'group_info', f'{tag}.csv'))

    mean_values = df[['Assigned_group', 'number_of_species_in_group']].groupby(
        'Assigned_group').mean()
    mean_values = mean_values.reset_index()

    def check_means(x):
        if x != int(x):
            raise ValueError
        else:
            pass

    mean_values['number_of_species_in_group'].apply(check_means)
    print(mean_values)

    # After mean values have been calculated, add compound data
    compound_data = pd.read_csv(all_species_compound_csv, index_col=0)
    working_data = pd.merge(compound_data, df, how='left', on='accepted_species', validate='many_to_many')
    working_data = working_data[working_data[wcvp_accepted_columns['family']].isin(FAMILIES_OF_INTEREST)]
    working_data = working_data.dropna(subset='Assigned_group')

    FAD_measures = calculate_FAD_measures(working_data, compound_grouping='Assigned_group')
    compiled_data = pd.merge(mean_values, FAD_measures, how='left', on='Assigned_group', validate='one_to_one')[['Assigned_group', div_metric]]
    compiled_data = compiled_data.rename(columns={'Assigned_group': 'Genus'})
    compiled_data = compiled_data.dropna(subset=div_metric)
    compiled_data.to_csv(os.path.join('outputs', 'group_data', f'{tag}.csv'))
    compiled_data.describe(include='all').to_csv(os.path.join('outputs', 'group_data', f'{tag}_summary.csv'))


def main():
    working_data = species_data.copy()

    working_data = working_data.rename(columns={'Genus': 'Assigned_group'})

    resolve_traits_to_group(
        working_data,
        tag='Genus')


def get_an_mcar_sample():
    df = pd.read_csv(os.path.join('outputs', 'group_data', 'Genus.csv'))[['Genus', 'APWD']]
    total_data = len(df)
    print(total_data)
    df.loc[df.sample(frac=0.1).index, 'APWD'] = np.nan
    nan_data = df[df['APWD'].isna()]
    print(len(nan_data))
    ratio = len(nan_data) / total_data
    print(ratio)
    assert ratio > 0.0999 and ratio < 0.1001
    df.to_csv(os.path.join('outputs', 'group_data', 'continuous_gentianales_mcar_sample.csv'), index=False)


if __name__ == '__main__':
    species_data = pd.read_csv(species_in_study_csv, index_col=0)[
        ['accepted_species', 'Genus']]

    # main()
    get_an_mcar_sample()
