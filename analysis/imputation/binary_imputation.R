library(TDIP)


method = "pi_categorical_traits"
pidisc = pi_categorical_traits(phyloNa_values$`PhyloNaN/5/0.05`,simData$TreeList$`1`)

errors <- imputation_error(imputedData = pidisc$imputedData,
                           trueData = simData$FinalData,
                           missingData = phyloNa_values$`PhyloNaN/5/0.05`,
                           imputationApproachesName = method)
x=pidisc$probabilities
