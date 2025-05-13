library(phytools)
library(ape)

source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')
repo_path = Sys.getenv('KEWSCRATCHPATH')
species_tree = ape::read.tree(file.path(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs',
                                    'Species', 'Uphylomaker_species_tree.tre'))
binary_data = read.csv(file.path('real_data', 'binary', 'binary_gentianales.csv'))
binary_data['accepted_species'] <- lapply(binary_data['accepted_species'], replace_space_with_underscore_in_name)

binary_cases <- function(){
  sample_tips = sample(species_tree$tip.label,param_tree[[3]])
  sample_tree = subset_tree(species_tree,sample_tips)
  ground_truth = binary_data[binary_data$accepted_species %in% sample_tips,]
  rownames(ground_truth) <- ground_truth$accepted_species
  ground_truth <- subset(ground_truth, select = -c(accepted_species))
  return(list(tree=sample_tree, FinalData= ground_truth, Dataframe=data.frame()))
}

for(i in 1:number_of_repetitions){
  binary_sample = binary_cases()
  ape::is.ultrametric(binary_sample$tree)
  output_simulation(file.path('real_data'),binary_sample, binary_sample$tree,'binary', i)
}

# From rBIEN package
# Maitner, Brian S., Brad Boyle, Nathan Casler, Rick Condit, John Donoghue, Sandra M. Durán, Daniel Guaderrama, et al. ‘The bien r Package: A Tool to Access the Botanical Information and Ecology Network (BIEN) Database’. Edited by Sean McMahon. Methods in Ecology and Evolution 9, no. 2 (February 2018): 373–79. https://doi.org/10.1111/2041-210X.12861.


# BIEN::BIEN_trait_list()
FAMILIES_OF_INTEREST = c('Gelsemiaceae', 'Gentianaceae', 'Apocynaceae', 'Loganiaceae', 'Rubiaceae')
cont_df = data.frame()
for (fam in FAMILIES_OF_INTEREST) {
  continuous_trait = BIEN::BIEN_trait_traitbyfamily(family=fam, trait='seed mass')
  cont_df = rbind(cont_df,continuous_trait)
}

## Tidy the collected data a bit
clean_df = cont_df[c("scrubbed_species_binomial", 'trait_value', 'unit')]
clean_df = clean_df[!is.na(clean_df$trait_value),]
clean_df$trait_value <- as.numeric(as.character(clean_df$trait_value))
clean_df = clean_df[clean_df$trait_value<1000,] ## Some clearly wrong records
clean_df = clean_df[clean_df$trait_value!=0,]
clean_df = aggregate(clean_df[, 2], list(clean_df$scrubbed_species_binomial), mean)
colnames(clean_df) = c('accepted_species', 'trait_value')
clean_df$trait_value = scale(clean_df$trait_value)
clean_df['accepted_species'] <- lapply(clean_df['accepted_species'], replace_space_with_underscore_in_name)

continuous_tree  = subset_tree(species_tree,clean_df$accepted_species)

continuous_cases <- function(){
  sample_tips = sample(continuous_tree$tip.label,param_tree[[3]])
  sample_tree = subset_tree(continuous_tree,sample_tips)
  ground_truth = clean_df[clean_df$accepted_species %in% sample_tips,]
  rownames(ground_truth) <- ground_truth$accepted_species
  ground_truth <- subset(ground_truth, select = -c(accepted_species))
  return(list(tree=sample_tree, FinalData= ground_truth, Dataframe=data.frame()))
}

for(i in 1:number_of_repetitions){
  cont_sample = continuous_cases()
  ape::is.ultrametric(cont_sample$tree)
  output_simulation(file.path('real_data'),cont_sample, cont_sample$tree,'continuous', i)
}
