source('helper_simulation_methods.R')
library(phytools)

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
  
  return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
  
}

get_OU_sample <- function(simulation_path){
    tree = ape::read.tree(file.path(simulation_path, 'tree.tre'))
    # https://revbayes.github.io/tutorials/cont_traits/simple_ou.html
    Alpha <- alpha <- exp(runif(1, log(0.5), log(2))) # The character is pulled toward the optimum by the rate of adaptation, α
    Theta <- runif(1, min = -10, max = 10) # a continuous character is assumed to evolve toward an optimal value, θ
    
    #https://blog.phytools.org/2013/11/new-ou-simulator-in-fastbm.html 
    trait_OU <- fastBM(tree, sig2=1, a=0, alpha=Alpha, theta=Theta)  # sig2 = BM variance, mu = trend strength
    trait_OU_scaled = scale(trait_OU)
    names(trait_OU_scaled) <- names(trait_OU)
    
    ground_truth = data.frame(trait_OU_scaled)
    
    param_dataframe = data.frame(Alpha=c(Alpha), Theta=c(Theta))
    
    return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))

}

get_BM_T_sample <- function(simulation_path, with_trend){
  tree = ape::read.tree(file.path(simulation_path, 'tree.tre'))
  if(with_trend){
  # Brownian Motion with a Trend (BM + Trend)
  mu = runif(1, min=-1, max=1)

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
  print('########## mu, min max')
  print(mu)
  print(min)
  print(max)
  print('##########')
  return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
}

get_binary_BMT_sample <- function(simulation_path){
  cont_example = get_BM_T_sample(simulation_path, TRUE)

  df = cont_example$FinalData
  min = min(df$trait_BM_trend_scaled)
  max = max(df$trait_BM_trend_scaled)

  threshold = runif(n=1, min=min, max=max)
  df['trait_BM_trend_scaled'] <- +(df$trait_BM_trend_scaled > threshold)

  return(list(tree=cont_example$tree, FinalData= df, Dataframe=cont_example$Dataframe))
}

for(i in 1:number_of_repetitions){
  print(i)
  for(case in c('ultrametric', 'with_extinct')){
    sim_path = file.path("trees", case, 'standard', i)   
    bmt_binary_sample = get_binary_BMT_sample(sim_path)
  }
}

