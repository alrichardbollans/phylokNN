library(phytools)
library(ape)

source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')

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
  
  ground_truth = data.frame(trait_BM_trend_scaled)
  
  param_dataframe = data.frame(mu=c(mu))
  return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
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
  
  ground_truth = data.frame(trait_EB_scaled)
  
  
  param_dataframe = data.frame(beta=c(beta))
  # phenogram(tree, trait_EB_scaled, fsize=0.8, main="Early Burst Model")
  
  
  return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
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
  
  
  
  if (!is.null(tree) && class(tree) == "phylo") {
    param_dataframe = data.frame(pars=c(pars))
    ground_truth = data.frame(traits)
    return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
  } else {
    get_bisse_sample()
  }
}

get_hisse_sample <- function(){
  # Heterogeneous Transition Rate Models
  # 
  # What if different clades evolve at different rates? {hisse} can vary transition rates across different lineages.
  # Example: Two rate classes (fast vs. slow evolving groups)
  
  turnover.rates = runif(2, min=0, max=1)
  eps.values= runif(2, min=0, max=1)
  transition.rates= matrix(runif(4, min=0, max=1), nrow=2)
  diag(transition.rates) <- NA
  
  simulated.result <- hisse::SimulateHisse(turnover.rates, eps.values, 
                                           transition.rates, max.taxa=param_tree[[3]], x0=0)
  hisse_tree = hisse::SimToPhylo(simulated.result, include.extinct=FALSE, drop.stem=TRUE)
  
  
  # # Define colors for binary states
  # trait_colors <- ifelse(traits == 1, "red", "blue")
  # 
  # # Plot tree with colored tip labels
  # plot(hisse_tree, tip.color = trait_colors, cex = 1.2)
  
  if (!is.null(hisse_tree) && class(hisse_tree) == "phylo") {
    traits = hisse_tree$tip.state
    traits = traits[match(hisse_tree$tip.label, names(traits))]
    ground_truth = data.frame(traits)
    param_dataframe = data.frame(turnover.rates=c(turnover.rates),eps.values=c(eps.values), transition.rates=c(transition.rates))
    return(list(tree=hisse_tree, FinalData= ground_truth, Dataframe=param_dataframe))
  } else {
    get_bisse_sample()
  }
  
  
}

for(i in 1:number_of_repetitions){
  BMT_sample = get_BMT_sample()
  output_simulation(file.path('non_standard_simulations','BMT'),BMT_sample, BMT_sample$tree,'continuous', i)
  
  EB_sample = get_EB_sample()
  output_simulation(file.path('non_standard_simulations','EB'),EB_sample, EB_sample$tree,'continuous', i)
  
  bisse_sample = get_bisse_sample()
  output_simulation(file.path('non_standard_simulations','BISSE'),bisse_sample, bisse_sample$tree,'binary', i)
  
  hisse_sample = get_hisse_sample()
  output_simulation(file.path('non_standard_simulations','HISSE'),hisse_sample, hisse_sample$tree,'binary', i)
}



