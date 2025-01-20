# Do binary and continuous cases under different evolutionary assumptions
# https://github.com/Matgend/TDIP
# install.packages(c('mlr3pipelines', 'missMDA', 'mlr3learners', 'Amelia', 'softImpute', 'missRanger'))
# install.packages(c('missForest', 'VIM'))
# install.packages("NADIA_0.4.2.tar.gz", repos = NULL, type = "source")
# install.packages(c('truncnorm'))
# install.packages("faux_1.2.1.tar.gz", repos = NULL, type = "source")
# remotes::install_version("geiger", version = "2.0.10") # see https://github.com/Matgend/TDIP/issues/1
# devtools::install_github("Matgend/TDIP")

# Gendre Matthieu. 2022. TDIP: Trait Data Imputation with Phylogeny.
library(TDIP)
# save distances, traits and trees to simulation folder, for phylonn to use


# The below redefines the rescaleTree function from TDIP to allow rescaling with both lambda and kappa
unlockBinding("rescaleTree", asNamespace("TDIP"))
rescaleTree <- function(tree, subdata){
  
  #rescale phylogeny applying lambda transformation BEFORE kappa.
  lambdaCheck <- mean(subdata$lambda)
  kappaCheck <- mean(subdata$kappa)
  subdataTree <- tree
  if(lambdaCheck != 1){
    subdataTree <- geiger::rescale(subdataTree, "lambda", lambdaCheck)
  }
  
  if (kappaCheck != 1){
    subdataTree <- geiger::rescale(subdataTree, "kappa", kappaCheck)
  }
  
  return(subdataTree)
}
assign("rescaleTree", rescaleTree, envir = asNamespace("TDIP"))
lockBinding("rescaleTree", asNamespace("TDIP"))

output_simulation <- function(simData, tree, tag,id, param_df){
  #PhyloNa
  phyloNa_values <- phyloNa_miss_meca(missingRate = missingRate,
                                      ds = simData$FinalData,
                                      tree = tree)
  #MCAR
  mcar_values <- mcar_miss_meca(missingRate = missingRate,
                                ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))
  
  ## Save data
  this_sim_path = file.path('simulations', tag, id)
  dir.create(this_sim_path)
  
  tree_distances = ape::cophenetic.phylo(tree)
  write.csv(tree_distances, file = file.path(this_sim_path, 'tree_distances.csv'))
  
  saveRDS(simData, file=file.path(this_sim_path, 'simData.rds'))
  
  write.csv(simData$FinalData, file.path(this_sim_path, 'simData_FinalData.csv'))
  write.csv(mcar_values, file.path(this_sim_path, 'mcar_values.csv'))
  saveRDS(mcar_values, file=file.path(this_sim_path, 'mcar_values.rds'))
  write.csv(phyloNa_values, file.path(this_sim_path, 'phyloNa_values.csv'))
  write.csv(param_df, file.path(this_sim_path, 'dataframe_params.csv'))
  write.csv(param_tree, file.path(this_sim_path, 'param_tree.csv'))
}
param_tree <- list(0.4, 0.1, 500)
missingRate <- 0.1
for(i in 1:10){
  ev_models = c('BM1', 'OU1')
  ev_model = ev_models[[sample(1:length(ev_models), 1)]]
  lambda = runif(1, min=0, max=1)
  kappa = runif(1, min=0, max=1)
  
  
  continuous_dataframe = data.frame(nbr_traits=c(1),
                                    class=c('continuous'),
                                    model = c(ev_model),
                                    states = c(1),
                                    correlation = c(1),
                                    uncorr_traits = c(1),
                                    fraction_uncorr_traits = c(0),
                                    lambda = c(lambda),
                                    kappa = c(kappa),
                                    highCor = c(0),
                                    manTrait = c(0))
  simcontinuousData <- data_simulator(param_tree = param_tree, 
                                      dataframe = continuous_dataframe)
  target_name = names(simcontinuousData$FinalData)
  # Z-score standardization for the "target" column
  simcontinuousData$FinalData[target_name] <- scale(simcontinuousData$FinalData[target_name])
  
  tree = simcontinuousData$TreeList$`1`
  output_simulation(simcontinuousData, tree,'continuous', i, continuous_dataframe)
  
}
for(i in 1:10){
  ev_models = c('ARD', 'SYM', 'ER')
  ev_model = ev_models[[sample(1:length(ev_models), 1)]]
  lambda = runif(1, min=0, max=1)
  kappa = runif(1, min=0, max=1)
  binary_dataframe = data.frame(nbr_traits=c(1),
                                    class=c('ordinal'),
                                    model = c(ev_model),
                                    states = c(2),
                                    correlation = c(1),
                                    uncorr_traits = c(1),
                                    fraction_uncorr_traits = c(0),
                                    lambda = c(lambda),
                                    kappa = c(kappa),
                                    highCor = c(0),
                                    manTrait = c(0))
  
  simbinaryData <- data_simulator(param_tree = param_tree, 
                                      dataframe = binary_dataframe)
  tree = simbinaryData$TreeList$`1`
  output_simulation(simbinaryData, tree,'binary', i,binary_dataframe)
}

