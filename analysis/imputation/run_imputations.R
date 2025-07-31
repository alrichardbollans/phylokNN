library(doParallel)
library(foreach)

num_cores <- detectCores() - 2  # Use all but one core
cl <- makeCluster(num_cores)
registerDoParallel(cl)

number_of_simulation_iterations = 100
# foreach(iter = 1:number_of_simulation_iterations) %dopar% {
for(iter in 1:number_of_simulation_iterations){
  repo_path = Sys.getenv('KEWSCRATCHPATH')
  source(file.path(repo_path, 'phyloKNN', 'analysis', 'imputation','R_continuous_imputation_helper_functions.R'))
  
  missingness_types = c('mcar', 'phyloNa')
  cases = c('ultrametric', 'with_extinct')
  binary_ev_models = c('ER', 'ARD', 'BiSSE', 'HiSSE', 'BMT*', 'MPNS')
  continuous_ev_models = c('BM', 'OU', 'EB', 'LB', 'BMT', 'BIEN')
  print(iter)
  for (missing_type in missingness_types) {
    for(ev_model in binary_ev_models){# Keep the inner loop sequential
        if(ev_model == 'MPNS'){
            run_picante_models('ultrametric', ev_model, iter, missing_type, 'binary')
            run_corHMM_models('ultrametric', ev_model, iter, missing_type, 'binary')
        }
        else{
            for(case in cases){
                run_picante_models(case, ev_model, iter, missing_type, 'binary')
                run_corHMM_models(case, ev_model, iter, missing_type, 'binary')
             }

        }
    }

     for(ev_model in continuous_ev_models){# Keep the inner loop sequential
        if(ev_model == 'BIEN'){
            run_phylopars_models('ultrametric', ev_model, iter, missing_type, 'continuous')
            run_picante_models('ultrametric', ev_model, iter, missing_type, 'continuous')
        }
        else{
            for(case in cases){
                run_phylopars_models(case, ev_model, iter, missing_type, 'continuous')
                run_picante_models(case, ev_model, iter, missing_type, 'continuous')
             }

        }

    }
    
  }
}
