repo_path <- Sys.getenv("KEWSCRATCHPATH")   
source('helpful_phyl_methods.R')


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


get_lambda_table <- function(df, tree, target, out_dir){
  data_with_tree_labels=get_matching_labels(tree,df)
  reordered_data = reorder_data_frame_like_tree(data_with_tree_labels,tree)
  # Following munkemuller_how_2012, use Pagel's lambda
  # phylogenetic imputation may perform poorly when lambda is less than 0.6 (Molina-Venegas et al., 2018)
  signal_lambda <- phytools::phylosig(tree, reordered_data[[target]], method="lambda",
                                    test=FALSE)
  
  lambda_table <-
  tribble(
    ~metric, ~value,
    "lambda",  signal_lambda$lambda,
  )

# save the table
write.csv(lambda_table, file.path(out_dir, "phylogenetic_signal_results_lambda.csv"))

}

get_D_table <- function(df, tree, target, out_dir){
  # D=1 suggests a phylogenetically random trait
  # D=0 a trait evolved under the Brownian model
  # pvalue1 giving the result of testing whether D is significantly different from one
  # pvalue0 giving the result of testing whether D is significantly different from zero
  
  if(length(unique(df[[target]])) == 1) {
    signal_D =''
    print('No signal as Unique target found')
  } else {
    names(df)[names(df) == target] <- 'target'
    
    # caper doesnt like having non unique node names
    if(length(tree$node.label[duplicated(tree$node.label)])>0){
      tree$node.label = as.character(seq(1, length(tree$node.label), length.out=length(tree$node.label)))
    }
    
    
    signal_D <- caper::phylo.d(df, tree, names.col=accepted_species, binvar=target)$DEstimate
  }
  
  
  lambda_table <-
    tribble(
      ~metric, ~value,
      "D",  signal_D,
    )

  # save the table
  write.csv(lambda_table, file.path(out_dir, "phylogenetic_signal_results_D.csv"))
  
}

number_of_simulation_iterations = 100

for(iter in 1:number_of_simulation_iterations){
  binary_ev_models = c('ER', 'ARD', 'BiSSE', 'HiSSE', 'bBMT', 'Clonality')
  continuous_ev_models = c('BM', 'OU', 'EB', 'LB', 'BMT', 'Seed Mass')
  print(iter)

  for(simulation_ev_model in c(continuous_ev_models, binary_ev_models)){
    
    if(simulation_ev_model == 'Clonality' | simulation_ev_model == 'Seed Mass'){
      cases=c('ultrametric')
    }
    else{
      cases = c('ultrametric', 'with_extinct')
    
    }
    for(case in cases){
      input_data = set_up(case, simulation_ev_model, iter, 'mcar')
      print(case)
      print(simulation_ev_model)
      print(iter)
      input_data_paths = get_input_data_paths(case, simulation_ev_model, iter)
      if( simulation_ev_model %in% continuous_ev_models){
        get_lambda_table(input_data$ground_truth, input_data$labelled_tree,input_data$target,input_data_paths$value_path )
      }
      else{
        get_D_table(input_data$ground_truth, input_data$labelled_tree,input_data$target,input_data_paths$value_path )
      }
    }
    
}

}
