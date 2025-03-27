repo_path = Sys.getenv('KEWSCRATCHPATH')
source(file.path(repo_path, 'phyloKNN', 'analysis', 'imputation','R_continuous_imputation_helper_functions.R'))
number_of_simulation_iterations = 100
missingness_types = c('mcar', 'phyloNa')


for (iter in c(1:number_of_simulation_iterations)) {
  print(iter)
  for(missing_type in missingness_types){
    
    # Binary cases
    run_picante_models('BISSE', 'binary', iter, missing_type)
    run_picante_models('HISSE', 'binary', iter, missing_type)
    run_corHMM_models('BISSE', 'binary', iter, missing_type)
    run_corHMM_models('HISSE', 'binary', iter, missing_type)
    
    # Continuous cases
    run_phylopars_models('BMT', 'continuous', iter, missing_type)
    run_phylopars_models('EB', 'continuous', iter, missing_type)
    
    run_picante_models('BMT', 'continuous', iter, missing_type)
    
    run_picante_models('EB', 'continuous', iter, missing_type)
  }
}
  
for (iter in c(2:27)){ #1:number_of_simulation_iterations)) {
  print(iter)
  for(missing_type in missingness_types){
    
    # run_picante_models('simulations', 'binary', iter, missing_type)
    # run_corHMM_models('simulations', 'binary', iter, missing_type)
    run_phylopars_models('simulations', 'continuous', iter, missing_type)
    run_picante_models('simulations', 'continuous', iter, missing_type)
  }
}




# With real data takes far too long
run_picante_models('real_data', 'binary', 1, 'mcar')
# run_corHMM_models('real_data', 'binary', 1, 'mcar')
# run_phylopars_models('real_data', 'continuous', 1, 'mcar')
# run_picante_models('real_data', 'continuous', 1, 'mcar')
print('Not implemented')
 
