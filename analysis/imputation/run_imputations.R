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
  
  



for(real_or_sim in c( 'real_data')){
  
    if(real_or_sim=='real_data'){
      # With real data takes far too long
      run_picante_models(real_or_sim, 'binary', 1, 'mcar')
      # run_corHMM_models(real_or_sim, 'binary', 1, 'mcar')
      # run_phylopars_models(real_or_sim, 'continuous', 1, 'mcar')
      # run_picante_models(real_or_sim, 'continuous', 1, 'mcar')
      print('Not implemented')
    }else{
      
        for (iter in c(1:number_of_simulation_iterations)) {
          print(iter)
          for(missing_type in missingness_types){
           
            run_picante_models(real_or_sim, 'binary', iter, missing_type)
            run_corHMM_models(real_or_sim, 'binary', iter, missing_type)
            run_phylopars_models(real_or_sim, 'continuous', iter, missing_type)
            run_picante_models(real_or_sim, 'continuous', iter, missing_type)
      }
    }
  }
  
}
