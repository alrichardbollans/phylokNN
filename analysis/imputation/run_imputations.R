repo_path = Sys.getenv('KEWSCRATCHPATH')
source(file.path(repo_path, 'phyloKNN', 'analysis', 'imputation','R_continuous_imputation_helper_functions.R'))
number_of_simulation_iterations = 100
missingness_types = c('mcar', 'phyloNa')

for(real_or_sim in c('simulations', 'real_data')){
  for(missing_type in missingness_types){
    if(real_or_sim=='real_data'){
      tune_binary_params(real_or_sim, 'binary', 1, missing_type)
      tune_continuous_params(real_or_sim, 'continuous', 1, missing_type)
    }else{
      for (iter in c(1:number_of_simulation_iterations)) {
        tune_binary_params(real_or_sim, 'binary', iter, missing_type)
        tune_continuous_params(real_or_sim, 'continuous', iter, missing_type)
      }
    }
  }
  
}
