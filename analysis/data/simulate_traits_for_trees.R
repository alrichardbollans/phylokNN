source('helper_simulation_methods.R')
library(phytools)

## Simulating traits for trees in given folders
## Outputs a named list containing the tree, ground_truth and a dataframe containing parameters for the trait simulation

get_ARD_or_ER_sample <- function(simulation_path, ev_model){
  tree = ape::read.tree(file.path(simulation_path, 'tree.tre'))
  
  if(ev_model== 'ARD'){
    rate <- runif(2, min = 0, max = 1)
  }
  if (ev_model== 'ER'){
    rate <- runif(1, min = 0, max = 1)
  }
  trait_ARD <- ape::rTraitDisc(tree, model=ev_model, k=2,rate=rate, states=c(0,1))
  
  ground_truth = data.frame(trait_ARD)
  
  param_dataframe = data.frame(rate=c(rate))
  out = list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe)
  output_simulation(simulation_path, out, ev_model)

}

get_OU_sample <- function(simulation_path){
    tree = ape::read.tree(file.path(simulation_path, 'tree.tre'))
    # https://revbayes.github.io/tutorials/cont_traits/simple_ou.html
    # alpha selection follows TDIP (https://github.com/Matgend/TDIP/blob/62c6655f7da66b0f89a48554a8eba7e697ea36eb/R/utils.R#L154)
    Alpha <- runif(1, 0.5, 2) # The character is pulled toward the optimum by the rate of adaptation, α
    # Theta selection follows TDIP (https://github.com/Matgend/TDIP/blob/62c6655f7da66b0f89a48554a8eba7e697ea36eb/R/data_simulator.R#L262)
    Theta <- runif(1, min = -10, max = 10) # a continuous character is assumed to evolve toward an optimal value, θ
    
    #https://blog.phytools.org/2013/11/new-ou-simulator-in-fastbm.html 
    trait_OU <- fastBM(tree, sig2=1, a=0, alpha=Alpha, theta=Theta)
    trait_OU_scaled = scale(trait_OU)
    names(trait_OU_scaled) <- names(trait_OU)
    
    ground_truth = data.frame(trait_OU_scaled)
    
    param_dataframe = data.frame(Alpha=c(Alpha), Theta=c(Theta))
    
    out = list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe)
  output_simulation(simulation_path, out, 'OU')


}

get_EBLB_sample <- function(simulation_path, early_or_late){
  tree = ape::read.tree(file.path(simulation_path, 'tree.tre'))
  # Early Burst (EB) Model
  #The early burst (EB) model assumes high rates of evolution early in a clade’s history that slow down or speed up over time.
  # It’s common in adaptive radiations.
  if(early_or_late=='LB'){
    r = runif(1, min=0, max=1)
  }
  if(early_or_late=='EB'){
    r = runif(1, min=-1, max=0)
  }
  print(r)
  
  # Explanation for this is here: https://www.biorxiv.org/content/10.1101/069518v1.full.pdf
  # and RPANDA.pdf
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
  out = list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe)
  output_simulation(simulation_path, out, early_or_late)

}

get_BM_T_sample <- function(simulation_path, with_trend){
  tree = ape::read.tree(file.path(simulation_path, 'tree.tre'))
  if(with_trend){
  # Brownian Motion with a Trend (BM + Trend)
  #Similar to Theta selection in TDIP (https://github.com/Matgend/TDIP/blob/62c6655f7da66b0f89a48554a8eba7e697ea36eb/R/data_simulator.R#L262)

    mu = runif(1, min=-10, max=10)
  }else{
    mu=0

  }

  trait_BM_trend <- fastBM(tree, sig2=1, a=0, mu=mu)  # sig2 = BM variance, mu = trend strength
  trait_BM_trend_scaled = scale(trait_BM_trend)
  names(trait_BM_trend_scaled) <- names(trait_BM_trend)
  # plot(trait_BM_trend_scaled, ylab="Trait Value", xlab="Species", main="BM with a Trend")
  # phenogram(tree, trait_BM_trend_scaled, fsize=0.8, main="Trait Evolution under BM with a Trend")

  ground_truth = data.frame(trait_BM_trend_scaled)

  param_dataframe = data.frame(mu=c(mu))

  min = min(ground_truth$trait_BM_trend_scaled)
  max = max(ground_truth$trait_BM_trend_scaled)
#   print('########## mu, min max')
#   print(mu)
#   print(min)
#   print(max)
#   print('##########')
  out = list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe)

  return(out)
}

get_binary_BMT_sample <- function(simulation_path){
  cont_example = get_BM_T_sample(simulation_path, TRUE)

  df = cont_example$FinalData
  min = min(df$trait_BM_trend_scaled)
  max = max(df$trait_BM_trend_scaled)

  threshold = runif(n=1, min=min, max=max)
  df['trait_BM_trend_scaled'] <- +(df$trait_BM_trend_scaled > threshold)

  out = list(tree=tree, FinalData= df, Dataframe=cont_example$Dataframe)
  output_simulation(simulation_path, out, 'bBMT')

}

for(i in 1:number_of_repetitions){
  print(i)
  for(case in c('ultrametric', 'with_extinct')){
    sim_path = file.path("simulations", case, 'standard', i)
    # binary cases
    get_ARD_or_ER_sample(sim_path,'ER')
    get_ARD_or_ER_sample(sim_path,'ARD')
    get_binary_BMT_sample(sim_path)

    # continuous cases
    get_OU_sample(sim_path)
    get_EBLB_sample(sim_path, 'EB')
    get_EBLB_sample(sim_path, 'LB')
    bm_sample = get_BM_T_sample(sim_path,FALSE)
    output_simulation(sim_path, bm_sample, 'BM')
    # 
    bmt_sample = get_BM_T_sample(sim_path,TRUE)
    output_simulation(sim_path, bm_sample, 'BMT')

  }
}

