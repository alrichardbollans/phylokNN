library(TDIP)
missingRate <- 0.1
number_of_repetitions = 100
number_of_taxa = 100
output_simulation <- function(this_sim_path, simData, tree, ev_model){
  param_df = simData$Dataframe
  extinct_tips = getExtinct(tree, tol=1e-8)
  #PhyloNa
  # the case in which species belonging to particular clades are more 
  # likely to be missing trait data
  phyloNa_values <- phyloNa_miss_meca(missingRate = missingRate,
                                      ds = simData$FinalData,
                                      tree = tree)[[1]]
  # Check some extinct tips are NaN where we have extinct tips
  if(length(extinct_tips)!=0){
    nan_tips = rownames(phyloNa_values)[which(is.na(phyloNa_values[, 1]))]
    extinct_and_nan = intersect(extinct_tips,nan_tips)
    while(length(extinct_and_nan)==0){
      phyloNa_values <- phyloNa_miss_meca(missingRate = missingRate,
                                          ds = simData$FinalData,
                                          tree = tree)[[1]]
      nan_tips = rownames(phyloNa_values)[which(is.na(phyloNa_values[, 1]))]
      extinct_and_nan = intersect(extinct_tips,nan_tips)
    }
  }
  #MCAR
  # Missing completely at random (MCAR), where a random sample of data 
  # independent of their values and other traits is missing
  mcar_values <- mcar_miss_meca(missingRate = missingRate,
                                ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))
  # Check some extinct tips are NaN where we have extinct tips
  if(length(extinct_tips)!=0){
    nan_tips = rownames(mcar_values)[which(is.na(mcar_values[, 1]))]
    extinct_and_nan = intersect(extinct_tips,nan_tips)
    while(length(extinct_and_nan)==0){
      mcar_values <- mcar_miss_meca(missingRate = missingRate,
                                    ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))
      nan_tips = rownames(mcar_values)[which(is.na(mcar_values[, 1]))]
      extinct_and_nan = intersect(extinct_tips,nan_tips)
    }
  }
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
  outpath = file.path(this_sim_path, ev_model)
  dir.create(outpath, recursive=TRUE)
  

  
  saveRDS(simData, file=file.path(outpath, 'simData.rds'))
  
  ground_truth = update_trait_columns(simData$FinalData)
  write.csv(ground_truth, file.path(outpath, 'ground_truth.csv'),row.names = FALSE)
  ## Write missing values
  write.csv(update_trait_columns(mcar_values), file.path(outpath, 'mcar_values.csv'),row.names = FALSE)
  saveRDS(mcar_values, file=file.path(outpath, 'mcar_values.rds'))
  
  # write.csv(update_trait_columns(mnar_values), file.path(outpath, 'mnar_values.csv'),row.names = FALSE)
  # saveRDS(mnar_values, file=file.path(outpath, 'mnar_values.rds'))
  
  # write.csv(update_trait_columns(mar_values), file.path(outpath, 'mar_values.csv'),row.names = FALSE)
  # saveRDS(mar_values, file=file.path(outpath, 'mar_values.rds'))
  
  write.csv(update_trait_columns(phyloNa_values), file.path(outpath, 'phyloNa_values.csv'),row.names = FALSE)
  saveRDS(phyloNa_values, file=file.path(outpath, 'phyloNa_values.rds'))
  
  write.csv(param_df, file.path(outpath, 'dataframe_params.csv'))
}


update_trait_columns <- function(df){
  df <- cbind(accepted_species = rownames(df), df)
  rownames(df) <- 1:nrow(df)
  return(df)
}
