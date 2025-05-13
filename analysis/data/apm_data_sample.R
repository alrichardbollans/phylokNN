## Only create one sample as computation is expensive with larger tree and methods work out similar to simulated study anyway?

library(phytools)
library(ape)

source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')
source('add_eigenvectors.R')
repo_path = Sys.getenv('KEWSCRATCHPATH')
species_tree = ape::read.tree(file.path(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs',
                                        'Species', 'Uphylomaker_species_tree.tre'))

# Data from APM Traits v1.12.1
binary_data = read.csv(file.path('my_apm_data', 'binary', 'compiled_extraction_apm_data.csv'))
binary_data['accepted_species'] <- lapply(binary_data['accepted_species'], replace_space_with_underscore_in_name)


sample_tips = binary_data$accepted_species
sample_tree = subset_tree(species_tree,sample_tips)
ground_truth = binary_data[binary_data$accepted_species %in% sample_tips,]
rownames(ground_truth) <- ground_truth$accepted_species
ground_truth <- subset(ground_truth, select = -c(accepted_species))
binary_sample = list(tree=sample_tree, FinalData= ground_truth, Dataframe=data.frame())
ape::is.ultrametric(binary_sample$tree)
output_simulation(file.path('my_apm_data'),binary_sample, binary_sample$tree,'binary', 1)

decompose_tree(file.path('my_apm_data','binary', 1))
