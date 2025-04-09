rm(list=ls())
library(tidyverse)
library(corrplot)
library(plotly)

###### Load the data ######
setwd("../Data")
sensory <- read_csv("final_ordered_wine_data_rf.csv") 

sensory_longer <- sensory %>%
  pivot_longer(
    ends_with("_avg"), 
    names_to = "attribute", 
    values_to = "avg", 
    values_drop_na = TRUE
  ) %>%
  mutate(
    attribute = str_sub(attribute, 1, -5),
    attribute_group = str_extract(attribute, "[^_]+$"),  # Extract the string after the last underscore
    attribute = str_remove(attribute, "_[^_]+$")  # Remove the last underscore and the following string
  ) %>%
  select(
    Wine_ID,
    attribute,
    attribute_group,
    avg
  )

###### Make initial eda ######
numeric_data <- sensory %>% 
  select_if(is.numeric)

# Calculate variance for each column
#col_vars <- apply(numeric_data, 2, var)

# Filter out columns with zero variance
numeric_data <- numeric_data %>% 
  select_if(~ var(.) > 0)

# Check if there are enough numeric columns left
if(ncol(numeric_data) > 1) {
  # Compute correlation matrix
  cor_matrix <- cor(numeric_data, use = "complete.obs")
  
  # Handle NA/NaN/Infs by removing these rows and columns
  cor_matrix <- cor_matrix[complete.cases(cor_matrix),]
  cor_matrix <- cor_matrix[, complete.cases(t(cor_matrix))]
  
  rownames(cor_matrix) <- str_remove_all(rownames(cor_matrix), "_avg")
  colnames(cor_matrix) <- str_remove_all(colnames(cor_matrix), "_avg")
  
  # Create corrplot
  corrplot(cor_matrix, 
           method = "color", 
           type = "lower", 
           order = "hclust", 
           tl.srt = 32.5, 
           tl.col = "black", 
           tl.cex = 0.72, 
           diag = FALSE)
} else {
  print("Not enough numeric columns with variance left for correlation analysis.")
}

