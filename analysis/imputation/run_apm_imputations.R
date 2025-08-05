library(doParallel)
library(foreach)

num_cores <- detectCores() - 2  # Use all but one core
cl <- makeCluster(num_cores)
registerDoParallel(cl)
case = 'ultrametric'
simulation_ev_model= 'APM'
number_of_simulation_iterations = 10
foreach(iter = 1:number_of_simulation_iterations) %dopar% {
  repo_path = Sys.getenv('KEWSCRATCHPATH')
  source(file.path(repo_path, 'phyloKNN', 'analysis', 'imputation','R_binary_imputation_helper_functions.R'))
  missingness_types = c('mcar')
  
  print(iter)
  for (missing_type in missingness_types) {  # Keep the inner loop sequential
    # Binary cases
    run_picante_models(case, simulation_ev_model, iteration, missing_type, bin_or_cont, 'AP')
    run_corHMM_models(case, simulation_ev_model, iteration, missing_type, bin_or_cont, 'AP')
    
  }
}
