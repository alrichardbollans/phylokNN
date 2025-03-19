
repo_path = Sys.getenv('KEWSCRATCHPATH')
source(file.path(repo_path, 'phyloKNN', 'analysis', 'imputation','R_binary_imputation_helper_functions.R'))
possible_phylopars_models = c('BM', 'mvOU','OU', "lambda", "kappa", "delta", "EB", "star")

format_phylopars <- function(phylopars_predictions,kfold_test_plants, target){
  plants_to_predict_that_are_in_tree = intersect(kfold_test_plants, rownames(phylopars_predictions))
  phylopars_predictions = phylopars_predictions[plants_to_predict_that_are_in_tree,target]
  output_data = data.frame(phylopars_predictions)
  
  
  output_data['estimate'] = output_data['phylopars_predictions']
  output_data = output_data[c('estimate')]
  # make index into column
  output_data <- cbind(accepted_species = rownames(output_data), output_data)
  rownames(output_data) <- 1:nrow(output_data)
  return(output_data)
}

tune_continuous_params <- function(real_or_sim, bin_or_cont, iteration, missing_type){
  setup_ = set_up(real_or_sim, bin_or_cont, iteration, missing_type)
  labelled_tree = setup_$labelled_tree
  if(!ape::is.ultrametric(labelled_tree)){
    labelled_tree = phytools::force.ultrametric(labelled_tree) # phylopars needs an ultrametric tree
  }
  missing_values_with_tree_labels = setup_$missing_values_with_tree_labels
  target = setup_$target
  non_missing_data = setup_$non_missing_data
  skfolds = setup_$skfolds
  training_tree = setup_$training_tree
  if(!ape::is.ultrametric(training_tree)){
    training_tree = phytools::force.ultrametric(training_tree) # phylopars needs an ultrametric tree
  }
  
  
  best_mae = -1
  best_ev_model = possible_phylopars_models[1]
  for (ev_model in possible_phylopars_models) {
    
    mae_for_this_config = 0
    for (i in 1:number_of_folds) {
      fold_indices <- skfolds[[i]]
      
      kfold_test_plants = non_missing_data[fold_indices,]$accepted_species
      
      test_data_with_tree_labels = data.frame(non_missing_data)
      # set some to unknown using NA
      test_data_with_tree_labels[[target]][test_data_with_tree_labels$label %in% kfold_test_plants] = NA
      
      phylopars_data = data.frame(test_data_with_tree_labels)
      colnames(phylopars_data)[1]  <- "species" #First column name of trait_data MUST be 'species' (all lower case).
      
      phylopars_data = subset(phylopars_data, select = c("species", target))
      p_v = Rphylopars::phylopars(phylopars_data, training_tree, model = ev_model)
      phylopars_predictions = p_v$anc_recon
      
      out = format_phylopars(phylopars_predictions, kfold_test_plants,target)
      validation_data = non_missing_data[non_missing_data$accepted_species %in% kfold_test_plants,]
      
      df_merge <- merge(out,validation_data,by="accepted_species")
      mae_for_this_fold = Metrics::mae(df_merge[[target]], df_merge$estimate)
      mae_for_this_config = mae_for_this_config+mae_for_this_fold
    }
    mae_for_this_config = mae_for_this_config/number_of_folds
    if (mae_for_this_config<best_mae || best_mae==-1){
      best_mae=mae_for_this_config
      best_ev_model = ev_model
    }
  }
  
  # Now use best model
  final_test_data_with_tree_labels = data.frame(missing_values_with_tree_labels)
  
  final_phylopars_data = data.frame(final_test_data_with_tree_labels)
  colnames(final_phylopars_data)[1]  <- "species" #First column name of trait_data MUST be 'species' (all lower case).
  
  final_phylopars_data = subset(final_phylopars_data, select = c("species", target))
  final_p_v = Rphylopars::phylopars(final_phylopars_data, labelled_tree, model = best_ev_model)
  final_phylopars_predictions = final_p_v$anc_recon
  
  plants_to_predict = final_phylopars_data[is.na(final_phylopars_data[[target]]),]$species
  
  final_out = format_phylopars(final_phylopars_predictions, plants_to_predict,target)
  
  dir.create(get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, missing_type), recursive=TRUE)
  write.csv(final_out, file.path(get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, missing_type), 'phylopars.csv'), row.names = FALSE)
  
  param_df = data.frame(best_ev_model=c(best_ev_model))
  write.csv(param_df, file.path(get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, missing_type), 'phylopars_hparams.csv'), row.names = FALSE)
  
}
