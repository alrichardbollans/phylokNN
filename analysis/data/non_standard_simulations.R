library(phytools)
library(ape)

source('helpful_phyl_methods.R')

get_tree <- function(){

  tree <- pbtree(n=param_tree[3])  # Simulate a tree with 50 taxa
  # plot(tree)
  return(tree)
}

get_BMT_sample <- function(){
  tree = get_tree()
  # Brownian Motion with a Trend (BM + Trend)
  mu = runif(1, min=-1, max=1)
  trait_BM_trend <- fastBM(tree, sig2=1, a=0, mu=mu)  # sig2 = BM variance, mu = trend strength
  trait_BM_trend_scaled = scale(trait_BM_trend)
  names(trait_BM_trend_scaled) <- names(trait_BM_trend)
  # plot(trait_BM_trend_scaled, ylab="Trait Value", xlab="Species", main="BM with a Trend")
  # phenogram(tree, trait_BM_trend_scaled, fsize=0.8, main="Trait Evolution under BM with a Trend")
  return(list(tree=tree, traits= trait_BM_trend_scaled))
}

get_EB_sample <- function(){
  tree = get_tree()
  # Early Burst (EB) Model
  #The early burst (EB) model assumes high rates of evolution early in a clade’s history that slow down over time. 
  # It’s common in adaptive radiations.
  beta = runif(1, min=-1, max=1)
  # beta < 0: Evolutionary rate declines over time.
  # beta > 0: Evolutionary rate increases over time.
  # beta = 0: Standard BM.
  trait_EB <- fastBM(tree, sig2=1, a=0, mu=0, beta=beta)
  trait_EB_scaled = scale(trait_EB)
  names(trait_EB_scaled) <- names(trait_EB)
  phenogram(tree, trait_EB_scaled, fsize=0.8, main="Early Burst Model")
  return(list(tree=tree, traits= trait_EB_scaled))
}

get_bisse_sample <- function(){
  # BiSSE (Binary-State Speciation and Extinction)
  # 
  # The BiSSE model (in {diversitree}) links a binary trait (0 or 1) to different birth/death rates.
  
  # Define birth/death rates depending on trait state
  
  # Define parameters: λ (speciation), μ (extinction), q01/q10 (transition rates)
  pars = runif(6, min=0.00001, max=0.99999)
  
  # Simulate a tree and associated binary trait
  # https://lukejharmon.github.io/ilhabela/2015/07/05/BiSSE-and-HiSSE/
  sim_bisse <- diversitree::tree.bisse(pars, max.taxa=param_tree[3], x0=0)
  
  # Visualize on tree
  # Extract the tree and traits
  tree <- sim_bisse
  traits <- sim_bisse$tip.state
  traits
  # Plot the tree with binary trait states
  if (!is.null(tree) && class(tree) == "phylo") {
    return(list(tree=tree, traits= traits))
  } else {
    get_bisse_sample()
  }
}

get_hisse_sample <- function(){
  # Heterogeneous Transition Rate Models
  # 
  # What if different clades evolve at different rates? {hisse} can vary transition rates across different lineages.
  # Example: Two rate classes (fast vs. slow evolving groups)

  simulated.result <- SimulateHisse(c(.3, .1), c(.1, 0), 
                                    matrix(c(NA, 0.2, .3, NA), nrow=2), max.taxa=param_tree[3], x0=0)
  hisse_tree = SimToPhylo(simulated.result, include.extinct=FALSE, drop.stem=TRUE)
  stop('Need to implement parameter randomisation')
  return(list(tree=hisse_tree, traits= hisse_tree$tip.state))
}

x= get_hisse_sample()

#PhyloNa
# the case in which species belonging to particular clades are more 
# likely to be missing trait data
missingRate = 0.1
phyloNa_values <- phyloNa_miss_meca(missingRate = missingRate,
                                    ds = trait_BM_trend_scaled,
                                    tree = tree)[[1]]
#MCAR
# Missing completely at random (MCAR), where a random sample of data 
# independent of their values and other traits is missing
mcar_values <- mcar_miss_meca(missingRate = missingRate,
                              ds = data.frame(trait_BM_trend_scaled),1)[[1]]
names(mcar_values) <- names(trait_BM_trend_scaled)






