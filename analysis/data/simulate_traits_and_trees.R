# Do binary and continuous cases under different evolutionary assumptions
# https://github.com/Matgend/TDIP
# install.packages('nloptr')
# install.packages(c('mlr3pipelines', 'missMDA', 'mlr3learners', 'Amelia', 'softImpute', 'missRanger'))
# install.packages(c('missForest', 'VIM'))
# install.packages("NADIA_0.4.2.tar.gz", repos = NULL, type = "source") #https://github.com/Matgend/TDIP?tab=readme-ov-file#installation
# install.packages(c('truncnorm'))
# install.packages("faux_1.2.1.tar.gz", repos = NULL, type = "source") #https://github.com/Matgend/TDIP/issues/1
# remotes::install_version("geiger", version = "2.0.10") # see https://github.com/Matgend/TDIP/issues/1
# packageVersion("geiger")
# install.packages(c('gmp', 'Rmpfr', 'corHMM')) # apt-get install libgmp-dev libmpfr-dev
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

number_of_repetitions = 100
param_tree <- list(0.4, 0.1, 100) # Values used in Gendre paper
missingRate <- 0.1

dir.create('simulations')

update_trait_columns <- function(df){
  df <- cbind(accepted_species = rownames(df), df)
  rownames(df) <- 1:nrow(df)
  return(df)
}

output_simulation <- function(simData, tree, tag,id){
  param_df = simData$Dataframe
  #PhyloNa
  # the case in which species belonging to particular clades are more 
  # likely to be missing trait data
  phyloNa_values <- phyloNa_miss_meca(missingRate = missingRate,
                                      ds = simData$FinalData,
                                      tree = tree)[[1]]
  #MCAR
  # Missing completely at random (MCAR), where a random sample of data 
  # independent of their values and other traits is missing
  mcar_values <- mcar_miss_meca(missingRate = missingRate,
                                ds = simData$FinalData, cols_mis = 1:ncol(simData$FinalData))

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
  this_sim_path = file.path('simulations', tag, id)
  dir.create(this_sim_path)
  
  tree_distances = ape::cophenetic.phylo(tree)
  write.csv(tree_distances, file = file.path(this_sim_path, 'tree_distances.csv'))
  
  saveRDS(simData, file=file.path(this_sim_path, 'simData.rds'))
  
  write.csv(update_trait_columns(simData$FinalData), file.path(this_sim_path, 'ground_truth.csv'),row.names = FALSE)
  ## Write missing values
  write.csv(update_trait_columns(mcar_values), file.path(this_sim_path, 'mcar_values.csv'),row.names = FALSE)
  saveRDS(mcar_values, file=file.path(this_sim_path, 'mcar_values.rds'))
  
  # write.csv(update_trait_columns(mnar_values), file.path(this_sim_path, 'mnar_values.csv'),row.names = FALSE)
  # saveRDS(mnar_values, file=file.path(this_sim_path, 'mnar_values.rds'))
  
  # write.csv(update_trait_columns(mar_values), file.path(this_sim_path, 'mar_values.csv'),row.names = FALSE)
  # saveRDS(mar_values, file=file.path(this_sim_path, 'mar_values.rds'))
  
  write.csv(update_trait_columns(phyloNa_values), file.path(this_sim_path, 'phyloNa_values.csv'),row.names = FALSE)
  saveRDS(phyloNa_values, file=file.path(this_sim_path, 'phyloNa_values.rds'))
  
  write.csv(param_df, file.path(this_sim_path, 'dataframe_params.csv'))
  write.csv(param_tree, file.path(this_sim_path, 'param_tree.csv'))
  
  ape::write.tree(simData$TreeList$`1`, file.path(this_sim_path, 'tree.tre'))
}


generate_continuous_sample <- function(){
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
  out <- data_simulator(param_tree = param_tree, dataframe = continuous_dataframe)
  return(out)
}


generate_binary_sample <- function(){
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
  
  out <- data_simulator(param_tree = param_tree, 
                                  dataframe = binary_dataframe)
  return(out)
}


for(i in 1:number_of_repetitions){
  simcontinuousData = generate_continuous_sample()
  target_name = names(simcontinuousData$FinalData)
  # Z-score standardization for the "target" column
  # so that MAE is comparable across different sets
  simcontinuousData$FinalData[target_name] <- scale(simcontinuousData$FinalData[target_name])
  
  tree = simcontinuousData$TreeList$`1`
  dir.create(file.path('simulations', 'continuous'))
  output_simulation(simcontinuousData, tree,'continuous', i)
  
}
for(i in 1:number_of_repetitions){
  simbinaryData = generate_binary_sample()
  tree = simbinaryData$TreeList$`1`
  dir.create(file.path('simulations', 'binary'))
  output_simulation(simbinaryData, tree,'binary', i)
}

