library(dplyr)

number_of_repetitions = 100
param_tree <- list(0.4, 0.1, 100) # Values used in Gendre paper
missingRate <- 0.1

replace_space_with_underscore_in_name<- function(given_name){
  new = gsub(" ", "_",given_name)
  return(new)
}

replace_underscore_with_space_in_name<- function(given_tree_name){
  new = gsub("_", " ",given_tree_name)
  return(new)
}

set_labels_on_tree_to_acc_name<- function(tree){
  tree$tip.label = as.character(lapply(tree$tip.label,replace_underscore_with_space_in_name))
  return(tree)
}

get_matching_labels <- function(tree,data){
  # Gets data which appears in tree and appends 'label' column
  # First match by accepted names
  accepted_label_matches <-
    tree$tip.label %>%
    tibble::enframe(name=NULL, value="label")%>%
    rowwise()  %>%
    mutate(accepted_species=label)%>%
    right_join(
      data,
      by=c("accepted_species"="accepted_species")
    )
  
  matching_labels = accepted_label_matches$label
  
  data_with_tree_labels_no_nan = tidyr::drop_na(accepted_label_matches,'label')
  
  return(data_with_tree_labels_no_nan)
}


subset_tree <- function(tree, node_list) {
  drop_list <- tree$tip.label[! tree$tip.label %in% node_list]
  
  return(ape::drop.tip(tree, drop_list))
}

get_subset_of_tree_from_names <- function(tree, names_to_include){
  
  lab_data = data.frame(accepted_species=names_to_include)
  
  return(get_subset_of_tree_from_data(lab_data, tree))
}

get_subset_of_tree_from_data <- function(data, tree){
  
  lab_data = data.frame(data)
  
  # print(lab_data)
  labels = get_matching_labels(tree,lab_data)$label
  
  # drop all tips we haven't found matches for
  f_tree <- subset_tree(tree, labels)
  
  return(f_tree)
}