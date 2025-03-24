library(TDIP)

output_simulation <- function(base_output_path, simData, tree, tag,id){
  param_df = simData$Dataframe
  #PhyloNa
  # the case in which species belonging to particular clades are more 
  # likely to be missing trait data
  phyloNa_values <- phyloNa_miss_meca(missingRate = missingRate,
                                      ds = simData$FinalData,
                                      tree = tree)[[1]]
  #MCAR
  # Missing completely at random (MCAR), where a random sample of data 
  # independent of their values and other traits is missing
  mcar_values <- mcar_miss_meca(missingRate = missingRate,
                                ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))
  
  #MNAR
  # missing not at random (MNAR), where missing data are a non-random 
  # subset of the values that does not relate to other traits included 
  # by the researcher in the dataset 
  # (See https://dl.acm.org/doi/abs/10.1145/1015330.1015425)
  # Reading, https://github.com/Matgend/TDIP/blob/62c6655f7da66b0f89a48554a8eba7e697ea36eb/R/mnar_miss_meca.R,
  # and https://www.rdocumentation.org/packages/missMethods/versions/0.4.0/topics/delete_MNAR_censoring,
  # my understanding is that this is basing the sample selection on the target,
  # which isn't MNAR (10.1145/1015330.1015425)
  # mnar_values <- mnar_miss_meca(missingRate = missingRate,
  #                               ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))
  
  #MAR
  # missing at random (MAR), where the distribution of missing values in a trait
  # is related to the values in other traits included in the dataset
  # NOT USED IN THE CURRENT STUDY
  # mar_values <- mar_miss_meca(missingRate = missingRate,
  #                               ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))
  
  ## Save data
  this_sim_path = file.path(base_output_path, tag, id)
  dir.create(this_sim_path, recursive=TRUE)
  
  tree_distances = ape::cophenetic.phylo(tree)
  write.csv(tree_distances, file = file.path(this_sim_path, 'tree_distances.csv'))
  
  saveRDS(simData, file=file.path(this_sim_path, 'simData.rds'))
  
  write.csv(update_trait_columns(simData$FinalData), file.path(this_sim_path, 'ground_truth.csv'),row.names = FALSE)
  ## Write missing values
  write.csv(update_trait_columns(mcar_values), file.path(this_sim_path, 'mcar_values.csv'),row.names = FALSE)
  saveRDS(mcar_values, file=file.path(this_sim_path, 'mcar_values.rds'))
  
  # write.csv(update_trait_columns(mnar_values), file.path(this_sim_path, 'mnar_values.csv'),row.names = FALSE)
  # saveRDS(mnar_values, file=file.path(this_sim_path, 'mnar_values.rds'))
  
  # write.csv(update_trait_columns(mar_values), file.path(this_sim_path, 'mar_values.csv'),row.names = FALSE)
  # saveRDS(mar_values, file=file.path(this_sim_path, 'mar_values.rds'))
  
  write.csv(update_trait_columns(phyloNa_values), file.path(this_sim_path, 'phyloNa_values.csv'),row.names = FALSE)
  saveRDS(phyloNa_values, file=file.path(this_sim_path, 'phyloNa_values.rds'))
  
  write.csv(param_df, file.path(this_sim_path, 'dataframe_params.csv'))
  
  ape::write.tree(tree, file.path(this_sim_path, 'tree.tre'))
}


update_trait_columns <- function(df){
  df <- cbind(accepted_species = rownames(df), df)
  rownames(df) <- 1:nrow(df)
  return(df)
}
