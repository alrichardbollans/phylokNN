## Create 10 fold CV to replicate ApmPP strategy

library(phytools)
library(ape)

source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')
source('add_eigenvectors.R')
repo_path = Sys.getenv('KEWSCRATCHPATH')
species_tree = ape::read.tree(file.path(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs',
                                        'Species', 'Uphylomaker_species_tree.tre'))

# Data from APM Traits v1.12.1 (WCVP v12)
binary_data = read.csv(file.path('my_apm_data', 'binary', 'compiled_extraction_apm_data.csv'))
binary_data['accepted_species'] <- lapply(binary_data['accepted_species'], replace_space_with_underscore_in_name)

accepted_species = binary_data$accepted_species
useful_tree = subset_tree(species_tree,accepted_species)
tree_distances = ape::cophenetic.phylo(useful_tree)
binary_data = binary_data[binary_data$accepted_species %in% accepted_species,]
ground_truth = binary_data
# rownames(ground_truth) <- ground_truth$accepted_species
# ground_truth <- subset(ground_truth, select = -c(accepted_species))
# rownames(ground_truth) <- ground_truth$accepted_species
# ground_truth <- subset(ground_truth, select = -c(accepted_species))
# binary_sample = list(tree=sample_tree, FinalData= ground_truth, Dataframe=data.frame())
# ape::is.ultrametric(binary_sample$tree)
# output_simulation(file.path('my_apm_data'),binary_sample, binary_sample$tree,'binary', 1)


number_of_folds = 10
# This does unstratified sampling
skfolds = caret::createFolds(ground_truth[['APM.Activity']], k=number_of_folds)

output_apm_cv_data <- function(base_output_path, id, mcar_values){

  ## Save data
  this_sim_path = file.path(base_output_path, 'binary', id)
  dir.create(this_sim_path, recursive=TRUE)
  
  
  write.csv(tree_distances, file = file.path(this_sim_path, 'tree_distances.csv'))
  
  write.csv(ground_truth, file.path(this_sim_path, 'ground_truth.csv'),row.names = FALSE)
  ## Write missing values
  write.csv(mcar_values, file.path(this_sim_path, 'mcar_values.csv'),row.names = FALSE)
  saveRDS(mcar_values, file=file.path(this_sim_path, 'mcar_values.rds'))
  
  write.csv(data.frame(), file.path(this_sim_path, 'dataframe_params.csv'))
  
  ape::write.tree(useful_tree, file.path(this_sim_path, 'tree.tre'))
}


for (i in 1:number_of_folds) {
  fold_indices <- skfolds[[i]]
  cv_sample = ground_truth
  cv_sample[fold_indices, "APM.Activity"] <- NA
  output_apm_cv_data(file.path('my_apm_data'), i,cv_sample)
  decompose_tree(file.path('my_apm_data','binary', i))
}




