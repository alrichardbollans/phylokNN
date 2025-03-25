library(TDIP)

for(i in 1:1000){
  print(i)
  tag = as.character(i)
  sim_path = file.path('simulations', 'binary')
  simData = readRDS(file.path(sim_path, tag, 'simData.rds'))
  
  missingness_types = c('mcar', 'phyloNa')
  # easier to read as R object to preserve dtype
  mcar_values = readRDS(file.path(sim_path, tag, 'mcar_values.rds')) 
  phyloNa_values = readRDS(file.path(sim_path, tag, 'phyloNa_values.rds'), row.names = 1)

  
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