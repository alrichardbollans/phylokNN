import pandas as pd
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

WCVP_VERSION = '12'
FAMILIES_OF_INTEREST = ['Gelsemiaceae', 'Gentianaceae', 'Apocynaceae', 'Loganiaceae', 'Rubiaceae']
if __name__ == '__main__':
    data = pd.read_csv('CoRRE_categoricalTraitData_Nov2023.csv')
    data = data[data['family'].isin(FAMILIES_OF_INTEREST)]
    print(data.head(10))
    print(data.columns.tolist())
    traits = data['trait'].unique()
    print(traits)
    data = data[data['trait'] == 'clonal']
    data = data.dropna(subset=['family', 'species', 'trait_value'], how='any')
    data = data.drop_duplicates(subset=['species'])
    data = data[data['trait_value'].isin(['yes', 'no'])]
    data['trait_value'] = data['trait_value'].apply(lambda x: 1 if x == 'yes' else 0)

    accepted_data = get_accepted_info_from_names_in_column(data[['species', 'trait_value']], 'species', wcvp_version=WCVP_VERSION)
    accepted_data = accepted_data.dropna(subset=['accepted_species'])
    accepted_data = accepted_data.drop_duplicates(subset=['accepted_species'])
    accepted_data[['accepted_family', 'accepted_species', 'trait_value']].to_csv(f'clonality.csv', index=False)
    accepted_data[['accepted_family', 'accepted_species', 'trait_value']].describe(include='all').to_csv(f'clonality_summary.csv')
