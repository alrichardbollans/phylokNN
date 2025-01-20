library(TDIP)

for(i in 1:10){
  print(i)
  tag = as.character(i)
  sim_path = file.path('simulations', 'binary')
  mcar_values = readRDS(file.path(sim_path, tag, 'mcar_values.rds')) # easier to read as R object to preserve dtype
  # phyloNa_values = read.csv(file.path(sim_path, tag, 'phyloNa_values.csv'), row.names = 1)
  simData = readRDS(file.path(sim_path, tag, 'simData.rds'))
  
  method = "pi_categorical_traits"
  target_name = names(mcar_values)
  
  pidisc = pi_categorical_traits(mcar_values[target_name],simData$TreeList$`1`)
  
  
  outdata = pidisc$probabilities
  write.csv(outdata, file.path('..','imputation','predictions', 'binary', tag, 'corHMM.csv'))
  write.csv(pidisc$parameters, file.path('..','imputation','predictions', 'binary', tag, 'corHMM_hparams.csv'))

}
# errors <- imputation_error(imputedData = pidisc$imputedData,
#                            trueData = simData$FinalData,
#                            missingData = phyloNa_values$`PhyloNaN/5/0.05`,
#                            imputationApproachesName = method)
# x=pidisc$probabilities