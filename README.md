#  Phylogenetic Nearest Neighbours

A Python package translating the concept of k Nearest Neighbours to phylogenetic trees for predicting trait values.

## Usage

To install with pip, run:

`pip install git+https://github.com/alrichardbollans/phylokNN`


How to set up and run the model:

```python

import pandas as pd

from phylokNN import PhylNearestNeighbours

# A pandas DataFrame representing the distance matrix between instances. Indices and columns should be taxon names. Should include all train, test species and unknown species for predictions.
distance_matrix = pd.read_csv('tree_distances.csv')
# A boolean indicating whether to output binary classes or continuous estimates for final predictions
clf = True
# A float value for the ratio of the largest tree distance to use as maximum distance threshold.
ratio = 0.8
# A float value for Kappa used to modify branch lengths, similar to Pagel's Kappa
kappa = 1.2
# A boolean indicating how to impute values of tips which have no neighbours under the distance threshold to impute. If False, left as NaN. If true, assigned mean trait value from training data
fill_in_unknowns_with_mean = True

phylnn = PhylNearestNeighbours(distance_matrix, clf, ratio, kappa, fill_in_unknowns_with_mean)

# First check the distance matrix conforms to the requirements
phylnn.check_integrity_of_distance_matrix(distance_matrix)

# {array-like, sparse matrix} of shape (n_samples, n_features). The first column must be names of tips/taxa.
train_data_X = pd.read_csv('train_data.csv')[['names']]

# The target variable. If this is a series, the index should be the same as the list of names from train_data_X, else order is assumed to match train_data_X
train_data_y = pd.read_csv('train_data.csv')['target']

# Optional sample weights. If this is a series, the index should be the same as the list of names from train_data_X, else order is assumed to match train_data_X
sample_weights = pd.read_csv('train_data.csv')['weights']

# Fit to the training data
phylnn.fit(train_data_X, train_data_y, sample_weight=sample_weights)

# {array-like, sparse matrix} of shape (n_samples, n_features). The first column must be names of tips/taxa.
test_data = pd.read_csv('test_data.csv')[['names']]

predictions = phylnn.predict(test_data)        
```


## Licence

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg