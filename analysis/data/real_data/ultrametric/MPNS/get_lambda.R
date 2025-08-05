repo_path <- Sys.getenv("KEWSCRATCHPATH")   
source('helpful_phyl_methods.R')
tree = ape::read.tree(file.path(repo_path,'gentianales_trees','WCVP_12','Uphy',  'outputs', 'Species','Uphylomaker_species_tree.tre'))
traits = read.csv(file.path('real_data','binary','binary_gentianales.csv'))

prepared_tree = set_labels_on_tree_to_acc_name(tree)

labelled_tree = get_subset_of_tree_from_data(traits,prepared_tree)
data_with_tree_labels=get_matching_labels(labelled_tree,traits)
# 
# named_vector = split(data_with_tree_labels[['Medicinal']], data_with_tree_labels[['accepted_species']])

reorder_data_frame_like_tree<-function(df,tree){
  
  # Get the tip labels from the tree
  tip_labels <- tree$tip.label
  
  # Create a new data frame with all tip labels and their order
  new_data <- data.frame(accepted_species = tip_labels, row.names = NULL)
  
  # Join the new data frame with the original data
  # This will add rows for missing labels and preserve the order
  merged_data <- left_join(new_data, df, by = "accepted_species")
  
  if(!all(merged_data$accepted_species == tree$tip.label)){
    stop("Mismatch with data and tree labels.")
  }
  
  return(merged_data)
}

reordered_data = reorder_data_frame_like_tree(data_with_tree_labels,labelled_tree)

# example_data = traits[1:10,]
# example_tree = get_subset_of_tree_from_data(example_data,prepared_tree)
# reordered_example_data = reorder_data_frame_like_tree(example_data,example_tree)
# Following munkemuller_how_2012, use Pagel's lambda
# phylogenetic imputation may perform poorly when lambda is less than 0.6 (Molina-Venegas et al., 2018)
signal_lambda <- phytools::phylosig(labelled_tree, reordered_data[['Medicinal']], method="lambda", 
                                    test=FALSE, niter=1)

lambda_table <-
  tribble(
    ~metric, ~value, ~pvalue,
    "lambda",  signal_lambda$lambda, signal_lambda$P
  )

# save the table
write.csv(lambda_table, "phylogenetic_signal_results_lambda.csv")
