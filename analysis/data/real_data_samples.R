library(phytools)
library(ape)

source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')
repo_path = Sys.getenv('KEWSCRATCHPATH')
species_tree = ape::read.tree(file.path(repo_path, 'gentianales_trees', 'WCVP_12', 'Uphy', 'outputs',
                                    'Species', 'Uphylomaker_species_tree.tre'))

### Binary case
## Kimberly J. Komatsu et al., ‘CoRRE Trait Data: A Dataset of 17 Categorical and Continuous Traits for 4079 Grassland Species Worldwide’,
## Scientific Data 11, no. 1 (2024): 795, https://doi.org/10.1038/s41597-024-03637-x.

binary_data = read.csv(file.path('real_data', 'ultrametric','Clonality', 'clonality.csv'))
binary_data = binary_data[,c("accepted_species", "trait_value")]
binary_data['accepted_species'] <- lapply(binary_data['accepted_species'], replace_space_with_underscore_in_name)
binary_data = binary_data[binary_data$accepted_species %in% species_tree$tip.label, ]
binary_tree  = subset_tree(species_tree,binary_data$accepted_species)

binary_cases <- function(){
  sample_tips = sample(binary_tree$tip.label,number_of_taxa)
  sample_tree = subset_tree(binary_tree,sample_tips)
  ground_truth = binary_data[binary_data$accepted_species %in% sample_tips,]
  rownames(ground_truth) <- ground_truth$accepted_species
  ground_truth <- subset(ground_truth, select = -c(accepted_species))
  return(list(tree=sample_tree, FinalData= ground_truth, Dataframe=data.frame()))
}

for(i in 1:number_of_repetitions){
  binary_sample = binary_cases()
  ape::is.ultrametric(binary_sample$tree)
  sim_path = file.path("real_data",'ultrametric', 'Clonality', i)
  output_tree(sim_path, binary_sample$tree)
  output_simulation(sim_path,binary_sample, 'Clonality')
}

### Continuous case
# From rBIEN package
# Maitner, Brian S., Brad Boyle, Nathan Casler, Rick Condit, John Donoghue, Sandra M. Durán, Daniel Guaderrama, et al. ‘The bien r Package: A Tool to Access the Botanical Information and Ecology Network (BIEN) Database’. Edited by Sean McMahon. Methods in Ecology and Evolution 9, no. 2 (February 2018): 373–79. https://doi.org/10.1111/2041-210X.12861.


bien_trait_list = BIEN::BIEN_trait_list()
FAMILIES_OF_INTEREST = c('Gelsemiaceae', 'Gentianaceae', 'Apocynaceae', 'Loganiaceae', 'Rubiaceae')



cont_df = data.frame()
for (fam in FAMILIES_OF_INTEREST) {
  continuous_trait = BIEN::BIEN_trait_traitbyfamily(family=fam, trait='seed mass')
  cont_df = rbind(cont_df,continuous_trait)
}
to_cite = BIEN::BIEN_metadata_citation(trait.dataframe=cont_df, bibtex_file = file.path("real_data",'ultrametric', 'Seed Mass','bien.bib'), acknowledgement_file='bien.txt')
articles = to_cite$references

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
  sample_tips = sample(continuous_tree$tip.label,number_of_taxa)
  sample_tree = subset_tree(continuous_tree,sample_tips)
  ground_truth = clean_df[clean_df$accepted_species %in% sample_tips,]
  rownames(ground_truth) <- ground_truth$accepted_species
  ground_truth <- subset(ground_truth, select = -c(accepted_species))
  return(list(tree=sample_tree, FinalData= ground_truth, Dataframe=data.frame()))
}

for(i in 1:number_of_repetitions){
  cont_sample = continuous_cases()
  ape::is.ultrametric(cont_sample$tree)
  sim_path = file.path("real_data",'ultrametric', 'Seed Mass', i)
  output_tree(sim_path, cont_sample$tree)
  output_simulation(sim_path,cont_sample, 'Seed Mass')
}
