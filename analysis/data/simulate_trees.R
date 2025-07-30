source('add_eigenvectors.R')
source('helper_simulation_methods.R')


get_tree <- function(include.extinct, birth,death,number_of_extant_taxa){

  SimTree <- phytools::pbtree(b = birth, d = death, n = number_of_extant_taxa, extant.only = !include.extinct)
  if (!is.null(SimTree) && class(SimTree) == "phylo" && length(phytools::getExtant(SimTree))==number_of_extant_taxa && 
      length(SimTree$tip.label)<500) {
    
    return(list("tree" = SimTree, "birth" = birth))
  } else {
    # To avoid recursive stack overflow..
    get_tree(include.extinct, birth*1.1,death,number_of_extant_taxa)
  }
}

get_bhisse_sample <- function(hidden.traits,include.extinct, number_of_extant_taxa){
  # BISSE and HiSSE internally generate trees as the state influences the birth and death rates.
  
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
  transition.rates <- hisse::TransMatMakerHiSSE(hidden.traits =hidden.traits, cat.trans.vary = TRUE)
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
  
  if (!is.null(hisse_tree) && class(hisse_tree) == "phylo" && length(phytools::getExtant(hisse_tree))==number_of_extant_taxa
      && length(hisse_tree$tip.label)<5*number_of_extant_taxa) {
    
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
  print(i)
  for(case in c('ultrametric', 'with_extinct')){
    
    this_sim_path = file.path("simulations", case, 'standard', i)
    death_rate = runif(1, min = 0, max = 1)
    birth_rate = runif(1, min = 0, max = 1)

    if(case=='ultrametric'){
      a = get_tree(FALSE, birth_rate,death_rate, number_of_taxa)
      testit::assert(ape::is.ultrametric(a$tree))
      testit::assert(length(a$tree$tip.label)==100)
    }
    if (case=='with_extinct'){
      a = get_tree(TRUE, birth_rate,death_rate, number_of_taxa)
      testit::assert(!ape::is.ultrametric(a$tree))
      testit::assert(length(a$tree$tip.label)>100)
    }
    tree = a$tree
    birth = a$birth
    output_tree(this_sim_path, tree, birth, death_rate)
  }
  
  ultra_bisse_sample = get_bhisse_sample(0,FALSE,number_of_taxa)
  sim_path= file.path("simulations", 'ultrametric', 'BiSSE', i)
  output_tree(sim_path, ultra_bisse_sample$tree)
  output_simulation(sim_path, ultra_bisse_sample,  'BiSSE')
  
  ultra_hisse_sample = get_bhisse_sample(1,FALSE,number_of_taxa)
  sim_path= file.path("simulations", 'ultrametric', 'HiSSE', i)
  output_tree(sim_path, ultra_hisse_sample$tree)
  output_simulation(sim_path, ultra_hisse_sample, 'HiSSE')
  
  extinct_bisse_sample = get_bhisse_sample(0,TRUE,number_of_taxa)
  sim_path= file.path("simulations", 'with_extinct', 'BiSSE', i)
  output_tree(sim_path, extinct_bisse_sample$tree)
  output_simulation(sim_path, extinct_bisse_sample, 'BiSSE')
  
  extinct_hisse_sample = get_bhisse_sample(1,TRUE,number_of_taxa)
  sim_path= file.path("simulations", 'with_extinct', 'HiSSE', i)
  output_tree(sim_path, extinct_hisse_sample$tree)
  output_simulation(sim_path, extinct_hisse_sample, 'HiSSE')

  
  
}
