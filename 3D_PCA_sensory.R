# Load necessary libraries
library(tidyverse)
library(missMDA)
library(FactoMineR)
library(factoextra)
library(plotly)

# Load the data
sensory <- read_csv("final_ordered_wine_data_pca.csv")

# Check the structure of the data
str(sensory)

# Ensure the row counts are consistent
original_row_count <- nrow(sensory)
print(paste("Original row count: ", original_row_count))

# Select only numeric columns, excluding non-quantitative ones
numeric_data <- sensory %>%
  select_if(is.numeric)
non_numeric_data <- sensory %>%
  select(-where(is.numeric))

# Double-check row counts for consistency
numeric_data_row_count <- nrow(numeric_data)
print(paste("Numeric data row count: ", numeric_data_row_count))

if (original_row_count != numeric_data_row_count) {
  stop("Row count mismatch between original data and numeric data.")
}

# Find the optimal number of components for PCA imputation
ncp_optimal <- estim_ncpPCA(numeric_data, ncp.max = 5) # Adjust ncp.max as necessary
optimal_ncp <- ncp_optimal$ncp
print(paste("Optimal number of components:", optimal_ncp))

# Impute missing values using the optimal number of components
imputed_data <- imputePCA(numeric_data, ncp = optimal_ncp)
complete_data <- imputed_data$completeObs

# Check imputed data structure and size
print("Imputed data structure:")
str(complete_data)
print(paste("Imputed data row count: ", nrow(complete_data)))
print(paste("Imputed data column count: ", ncol(complete_data)))

if (nrow(complete_data) != original_row_count) {
  stop("Row count mismatch after imputation.")
}

# Perform PCA
pca_result <- PCA(complete_data, scale.unit = TRUE, ncp = 3, graph = FALSE)

# Validate PCA result
print("PCA result structure:")
str(pca_result)

# Access individual PCA coordinates for PCA results
pca_scores <- data.frame(pca_result$ind$coord)

# Print the PCA scores data frame to inspect its contents
print("PCA Scores:")
print(head(pca_scores))
print(paste("PCA scores row count: ", nrow(pca_scores)))

# Ensure the row count of pca_scores matches the original data
if (nrow(pca_scores) != original_row_count) {
  stop("Row count mismatch between PCA scores and original data.")
}

# Add Grape Cultivar back to PCA scores
pca_scores$Grape_Cultivar <- sensory$Grape_Cultivar

# Print the revised PCA scores data frame
print("PCA Scores with Grape Cultivar:")
print(head(pca_scores))

# 3D Visualization using plotly
fig <- plot_ly(
  data = pca_scores,
  x = ~Dim.1,
  y = ~Dim.2,
  z = ~Dim.3,
  color = ~Grape_Cultivar,
  colors = c('#E7B800', '#8E44AD', '#FC4E07'),
  type = 'scatter3d',
  mode = 'markers'
) %>% 
  layout(
    scene = list(
      xaxis = list(title = 'Dim.1'),
      yaxis = list(title = 'Dim.2'),
      zaxis = list(title = 'Dim.3')
    )
  )

# Show the plot
fig

# Tallying dimension variance by input factor.
sum(pca_result$var$contrib[1:length(pca_result$var$contrib)])

# I want to combine the input factors into color, aroma, taste, and other categories.
# Load necessary library for better data manipulation
library(dplyr)

# Extract contributions from PCA results
contribs <- pca_result$var$contrib

# Create a data frame for easier manipulation
contribs_df <- as.data.frame(contribs)

# Initialize sums for each category
aroma_sum <- 0
taste_sum <- 0
color_sum <- 0
other_sum <- 0

# Iterate through the contributions and classify
for (variable in rownames(contribs_df)) {
  if (grepl("aroma", tolower(variable))) {
    aroma_sum <- aroma_sum + sum(contribs_df[variable, ])
  } else if (grepl("taste", tolower(variable)) || grepl("intensity", tolower(variable))) {
    taste_sum <- taste_sum + sum(contribs_df[variable, ])
  } else if (grepl("color", tolower(variable))) {
    color_sum <- color_sum + sum(contribs_df[variable, ])
  } else {
    other_sum <- other_sum + sum(contribs_df[variable, ])
  }
}

# Print the results
cat("Total Contribution for Aroma:", aroma_sum, "\n")
cat("Total Contribution for Taste:", taste_sum, "\n")
cat("Total Contribution for Color:", color_sum, "\n")
cat("Total Contribution for Other:", other_sum, "\n")

# Define the function
top_contributions <- function(contrib_data, n = 5) {
  sorted_indices <- order(contrib_data, decreasing = TRUE)
  top_var_names <- rownames(contribs_df)[sorted_indices[1:n]]
  return(top_var_names)
}

# Print top contributing variables for each PCA dimension
cat("Top contributing variables for Dim.1:\n")
print(top_contributions(contribs_df[, "Dim.1"], n = 5))

cat("\nTop contributing variables for Dim.2:\n")
print(top_contributions(contribs_df[, "Dim.2"], n = 5))

cat("\nTop contributing variables for Dim.3:\n")
print(top_contributions(contribs_df[, "Dim.3"], n = 5))
