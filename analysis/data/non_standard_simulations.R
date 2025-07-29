library(phytools)
library(ape)

source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')

get_tree <- function(){

  tree <- pbtree(n=param_tree[3])
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
  #The early burst (EB) model assumes high rates of evolution early in a clade’s history that slow down or speed up over time. 
  # It’s common in adaptive radiations.
  r = runif(1, min=-1, max=1)
  # Explanation for this is here: https://www.biorxiv.org/content/10.1101/069518v1.full.pdf
  # and file:///home/atp/Downloads/RPANDA.pdf
  modelACDC = RPANDA::createModel(tree, 'ACDC')
  #method 3 Simulates step-by-step the whole trajectory, but returns only the tip data (to plot change to 2)
  dataACDC <- RPANDA::simulateTipData(modelACDC, c(0,0,1,r), method=3) #
  trait_EB_scaled = scale(dataACDC)
  names(trait_EB_scaled) <- names(dataACDC)
  # simulateTipData doesn't preserve tip order annoyingly, so fix this
  sorted = as.data.frame(trait_EB_scaled)[tree$tip.label,]
  names(sorted)=tree$tip.label
  ground_truth = data.frame(sorted)
  
  
  param_dataframe = data.frame(r=c(r))
  # phenogram(tree, sorted, fsize=0.8, main="Early Burst Model")
  
  
  return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
}

get_bhisse_sample <- function(hidden.traits,include.extinct, number_of_extant_taxa){
  
  if(hidden.traits==1){
    # Heterogeneous Transition Rate Models
    # https://revbayes.github.io/tutorials/sse/hisse
    # a HiSSE model with 1 hidden binary trait (2 hidden states) and 1 observed binary trait (2 observed states), totaling 4 states. 
    # This allows for interactions between hidden and observed traits in diversification rates.
    # For 4 states (e.g., 0A, 0B, 1A, 1B), define:
    death_rates = runif(4, min = 0, max = 1)
    birth_rates = runif(4, min = 0, max = 1)
    
  }
  if(hidden.traits==0){
    # BiSSE (Binary-State Speciation and Extinction)
    # https://revbayes.github.io/tutorials/sse/bisse-intro.html#bisse_theory
    # The BiSSE model (in {diversitree}) links a binary trait (0 or 1) to different birth/death rates.
    
    # Define birth/death rates depending on trait state
    death_rates = runif(2, min = 0, max = 1)
    birth_rates = runif(2, min = 0, max = 1)
  }
  
  turnover.rates <- death_rates+birth_rates  # λ + μ for each of 4 states
  eps.values <- death_rates/birth_rates     # μ/λ ratios for each state
  
  # Get indices for transition rates. Allow transition among hidden categories to vary.
  transition.rates <- TransMatMakerHiSSE(hidden.traits =hidden.traits, cat.trans.vary = TRUE)
  for(i in 1:6){
    transition.rates[transition.rates==i]<-runif(1, min = 0, max = 1)
  }
  simulated.result <- hisse::SimulateHisse(turnover.rates, eps.values, 
                                           transition.rates, max.taxa=number_of_extant_taxa, x0=0)
  hisse_tree = hisse::SimToPhylo(simulated.result, include.extinct=include.extinct, drop.stem=TRUE)
  # plot(hisse_tree)
  
  # # Define colors for binary states
  # trait_colors <- ifelse(traits == 1, "red", "blue")
  # 
  # # Plot tree with colored tip labels
  # plot(hisse_tree, tip.color = trait_colors, cex = 1.2)

  if (!is.null(hisse_tree) && class(hisse_tree) == "phylo" && length(getExtant(hisse_tree))==number_of_extant_taxa) {

    if(hidden.traits==1){
      # Convert states back into observed binary character
      # Extract tip states (0, 1, 2, 3)
      tip_states <- hisse_tree$tip.state
      
      # Map to observed binary trait (0 or 1)
      observed_traits <- tip_states %% 2  # 0,2 → 0, 1,3 → 1
      
      # Overwrite tip names (careful!)
      hisse_tree$tip.state <- observed_traits
    }
    
    traits = hisse_tree$tip.state
    traits = traits[match(hisse_tree$tip.label, names(traits))]
    ground_truth = data.frame(traits)
    param_dataframe = data.frame(turnover.rates=c(turnover.rates),eps.values=c(eps.values), transition.rates=c(transition.rates))
    return(list(tree=hisse_tree, FinalData= ground_truth, Dataframe=param_dataframe))
  } else {
    get_bhisse_sample(hidden.traits, include.extinct, number_of_extant_taxa)
  }
  
  
}

for(i in 1:number_of_repetitions){
  # BMT_sample = get_BMT_sample()
  # ape::is.ultrametric(BMT_sample$tree)
  # output_simulation(file.path('non_standard_simulations','BMT'),BMT_sample, BMT_sample$tree,'continuous', i)

  EB_sample = get_EB_sample()
  ape::is.ultrametric(EB_sample$tree)
  output_simulation(file.path('non_standard_simulations','EB'),EB_sample, EB_sample$tree,'continuous', i)

  bisse_sample = get_bhisse_sample(0, FALSE)
  exbisse_sample = get_bhisse_sample(0, TRUE)
  # ape::is.ultrametric(bisse_sample$tree)
  # output_simulation(file.path('non_standard_simulations','BISSE'),bisse_sample, bisse_sample$tree,'binary', i)
  # 
  hisse_sample = get_bhisse_sample(1, FALSE)
  exhisse_sample = get_bhisse_sample(1, TRUE)
  # ape::is.ultrametric(hisse_sample$tree)
  # output_simulation(file.path('non_standard_simulations','HISSE'),hisse_sample, hisse_sample$tree,'binary', i)
}
