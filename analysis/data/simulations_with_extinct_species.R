source('helpful_phyl_methods.R')
source('helper_simulation_methods.R')

library(phytools)
library(ape)

get_tree <- function(){
  
  tree <- pbtree(b=1,d=0.33,n=param_tree[3])
  while (length(getExtant(tree))!= param_tree[3]){
    tree <- pbtree(b=1,d=0.33,n=param_tree[3])
  }
  # plot(tree)
  # print(ape::is.ultrametric(tree))
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
  
  min = min(ground_truth$trait_BM_trend_scaled)
  max = max(ground_truth$trait_BM_trend_scaled)
  print('########## mu, min max')
  print(mu)
  print(min)
  print(max)
  print('##########')
  return(list(tree=tree, FinalData= ground_truth, Dataframe=param_dataframe))
}

get_binary_BMT_sample <- function(){
  cont_example = get_BMT_sample()
  
  df = cont_example$FinalData
  min = min(df$trait_BM_trend_scaled)
  max = max(df$trait_BM_trend_scaled)
  
  threshold = runif(n=1, min=min, max=max)
  df['trait_BM_trend_scaled'] <- +(df$trait_BM_trend_scaled > threshold)
  
  return(list(tree=cont_example$tree, FinalData= df, Dataframe=cont_example$Dataframe))
}

for(i in 1:number_of_repetitions){
  BMT_sample = get_BMT_sample()
  if(ape::is.ultrametric(BMT_sample$tree)){
    stop()
  }
  output_simulation(file.path('non_ultrametric_simulations','Extinct_BMT'),BMT_sample, BMT_sample$tree,'continuous', i)
  
  binary_BMT_sample = get_binary_BMT_sample()
  if(ape::is.ultrametric(binary_BMT_sample$tree)){
    stop()
  }
  output_simulation(file.path('non_ultrametric_simulations','Extinct_BMT'),binary_BMT_sample, binary_BMT_sample$tree,'binary', i)
  
}

