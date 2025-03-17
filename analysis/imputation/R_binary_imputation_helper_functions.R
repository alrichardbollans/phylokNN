
number_of_folds=5
possible_corr_ev_models = c('ARD', 'ER','SYM')
possible_rate_cats = c(1,2)

repo_path = Sys.getenv('KEWSCRATCHPATH')
input_data_path = file.path(repo_path, 'phyloKNN', 'analysis', 'data')
prediction_path = file.path(repo_path, 'phyloKNN', 'analysis', 'imputation')

get_iteration_path_from_base <- function(base, real_or_sim, bin_or_cont, iteration) {
  if (real_or_sim == "real_data" || real_or_sim == "simulations") {
    basepath <- file.path(base, real_or_sim)
  } else {
    stop("Unknown real or simulation data")
  }

  if (bin_or_cont == "binary" || bin_or_cont == "continuous") {
    nextpath <- file.path(basepath, bin_or_cont)
  } else {
    stop(paste("Unknown data type", bin_or_cont))
  }

  iterpath <- file.path(nextpath, as.character(iteration))

  return(iterpath)
}

get_input_data_paths <- function(real_or_sim, bin_or_cont, iteration) {
  return(get_iteration_path_from_base(input_data_path, real_or_sim, bin_or_cont, iteration))
}

get_prediction_data_paths <- function(real_or_sim, bin_or_cont, iteration, missingness_type) {
  return(file.path(get_iteration_path_from_base(prediction_path, real_or_sim, bin_or_cont, iteration), missingness_type))
}

# Methods to format PI outputs to standarise like phyestimatedisc output
format_corhmm <- function(corhmm_output, plant_names_to_predict, ratecat){
  # print('Formatting')
  # Extract tip states from corhmm output
  # This could be much cleaner. See https://github.com/thej022214/corHMM/issues/62
  output_data = data.frame(corhmm_output$tip.states)
  
  # Filter data for plants to predict
  plants_to_predict_that_are_in_tree = intersect(plant_names_to_predict, rownames(output_data))
  output_data = output_data[plants_to_predict_that_are_in_tree,]
  
  
  if (ratecat ==1){
    output_data['0']=output_data$X1
    output_data['1']= output_data$X2
  }
  if (ratecat ==2){
    output_data['0']=output_data$X1+output_data$X3
    output_data['1']= output_data$X2+output_data$X4
  }
  
  
  
  output_data['estimated.state'] = lapply(output_data['1'], round)
  return(output_data)
}

# real_or_sim = 'simulations'
# bin_or_cont = 'binary'
# iteration =1
# missing_type = 'mnar'
# rate_cat = 2
# ev_model = 'ER'
# i=1
tune_binary_params <- function(real_or_sim, bin_or_cont, iteration, missing_type){

    data_path = get_input_data_paths(real_or_sim, bin_or_cont, iteration)
    ground_truth = read.csv(file.path(data_path, 'ground_truth.csv'))
    missing_values = read.csv(file.path(data_path, paste(missing_type,'_values.csv',sep='')))
    
    
    if (real_or_sim == "real_data"){
      labelled_tree = ape::read.tree(file.path(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs', 
                                               'Species', 'Uphylomaker_species_tree.tre'))
    } else if(real_or_sim == "simulations") {
      labelled_tree = ape::read.tree(file.path(data_path, 'tree.tre'))
    }
    
    target = names(ground_truth)[2]
    non_missing_data = missing_values[!is.na(missing_values[[target]]),]
    skfolds = caret::createFolds(factor(non_missing_data[[target]]), k=number_of_folds)

    best_brier_score = 1
    best_rate_cat = possible_rate_cats[1]
    best_ev_model = possible_corr_ev_models[1]
    for (rate_cat in possible_rate_cats) {
      for (ev_model in possible_corr_ev_models) {
        

        for (i in 1:number_of_folds) {
            fold_indices <- skfolds[[i]]
            kfold_test_plants = non_missing_data[fold_indices,]$accepted_species
            
            test_data_with_tree_labels = data.frame(non_missing_data)
            # set unknown using '?' for corhmm
            # corhmm will estimate trait values for all tips in tree that are in trait data with ? value
            test_data_with_tree_labels[[target]][test_data_with_tree_labels$accepted_species %in% kfold_test_plants] = '?'
            cor_trait_data = data.frame(test_data_with_tree_labels)
            cor_trait_data = subset(cor_trait_data, select = c("accepted_species", target))
            
            predicted_values = corHMM::corHMM(labelled_tree, cor_trait_data,model=ev_model,
                                              rate.cat = rate_cat, get.tip.states = TRUE, n.cores = 10)
      
            out = format_corhmm(predicted_values, kfold_test_plants, rate_cat)
            
            out <- cbind(accepted_species = rownames(out), out)
            rownames(out) <- 1:nrow(out)
            
            validation_data = non_missing_data[non_missing_data$accepted_species %in% kfold_test_plants,]
            
            df_merge <- merge(out,validation_data,by="accepted_species") 
            
            f_t = df_merge$`1`
            o_t = as.numeric(df_merge[[target]])
            brier_score = mean((f_t - o_t)^2)
            
            if (brier_score<best_brier_score){
              best_brier_score=brier_score
              best_rate_cat = rate_cat
              best_ev_model = ev_model
            }
            
          }

}
    }
    
    asda this needs fixing for correct dataa by removing tips in tree that arent in data
    
    final_test_data_with_tree_labels = data.frame(missing_values)
    # set unknown using '?' for corhmm
    # corhmm will estimate trait values for all tips in tree that are in trait data with ? value
    final_test_data_with_tree_labels[[target]][is.na(final_test_data_with_tree_labels[[target]])] = '?'
    final_cor_trait_data = data.frame(final_test_data_with_tree_labels)
    final_cor_trait_data = subset(final_cor_trait_data, select = c("accepted_species", target))
    
    final_predicted_values = corHMM::corHMM(labelled_tree, final_cor_trait_data,model=best_ev_model,
                                      rate.cat = best_rate_cat, get.tip.states = TRUE, n.cores = 10)
    
    final_out = format_corhmm(predicted_values, kfold_test_plants, rate_cat)
}



