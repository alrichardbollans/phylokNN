library(TDIP)
# tag='1'

for(i in 1:10){
  tag = as.character(i)
  sim_path = file.path('simulations', 'continuous')
  mcar_values = read.csv(file.path(sim_path, tag, 'mcar_values.csv'), row.names = 1)
  simData = readRDS(file.path(sim_path, tag, 'simData.rds'))

  target_name = names(mcar_values)
  picont = pi_continuous_traits(mcar_values[target_name],simData$TreeList$`1`)
  outdata = picont$imputedData
  write.csv(outdata, file.path('..','imputation','predictions', 'continuous', tag, 'Rphylopars.csv'))
  
  write.csv(picont$parameters$model, file.path('..','imputation','predictions', 'continuous', tag, 'Rphylopars_hparams.csv'))
}


# method = "pi_continuous_traits"
# errors <- imputation_error(imputedData = picont$imputedData,
#                            trueData = simData$FinalData,
#                            missingData = phyloNa_values$`PhyloNaN/5/0.05`,
#                            imputationApproachesName = method)
# x=picont$imputedData
