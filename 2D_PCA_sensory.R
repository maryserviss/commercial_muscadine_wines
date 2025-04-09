# Load necessary libraries
library(tidyverse)
library(missMDA)
library(FactoMineR)
library(factoextra)

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
pca_result <- PCA(complete_data, scale.unit = TRUE, graph = FALSE)

# Validate PCA result
print("PCA result structure:")
print(pca_result$eig)
str(pca_result)

# Ensure that coordinates (x) from PCA result are correctly accessed
if (!is.null(pca_result$ind$coord)) {
  # Access individual coordinates for PCA results
  pca_scores <- data.frame(pca_result$ind$coord)
  
  pca_result_row_count <- nrow(pca_scores)
  print(paste("PCA result row count: ", pca_result_row_count))
  
  if (pca_result_row_count != original_row_count) {
    stop("Row count mismatch in PCA results.")
  }
  
  # Use 'Grape_Cultivar' column for coloring
  color_var <- sensory$Grape_Cultivar
  legend_title <- "Grape Cultivar"
  
  # Define a palette with enough colors for the Grape Cultivars
  palette <- c("#E7B800", "#8E44AD", "#FC4E07")
  
  # Visualize PCA
  pca_plot <- fviz_pca_ind(pca_result,
                           geom.ind = "point",
                           col.ind = color_var,
                           palette = palette,
                           addEllipses = TRUE,
                           legend.title = legend_title,
                           pointsize = 2,
                           alpha.ind = 1) +
    ggtitle("PCA Plot of Grape Cultivars")
  
  # Display the PCA plot
  print(pca_plot)
  
  # ---- Statistical analysis ----
  # Add Grape Cultivar back to PCA scores
  pca_scores$Grape_Cultivar <- sensory$Grape_Cultivar
  
  # Print the revised PCA scores data frame
  print("PCA Scores with Grape Cultivar:")
  print(head(pca_scores))
  
  # Perform ANOVA on the first two principal components
  anova_Dim1 <- aov(Dim.1 ~ Grape_Cultivar, data = pca_scores)
  anova_Dim2 <- aov(Dim.2 ~ Grape_Cultivar, data = pca_scores)
  anova_Dim3 <- aov(Dim.3 ~ Grape_Cultivar, data = pca_scores)
  
  # Output ANOVA summaries
  print("ANOVA for Dim.1:")
  summary_Dim1 <- summary(anova_Dim1)
  print(summary_Dim1)
  
  print("ANOVA for Dim.2:")
  summary_Dim2 <- summary(anova_Dim2)
  print(summary_Dim2)
  
  print("ANOVA for Dim.3:")
  summary_Dim3 <- summary(anova_Dim3)
  print(summary_Dim3)
} else {
  print("PCA result 'ind$coord' is NULL. Unable to proceed with further analysis.")
}

## Interpreting results:
## Statistically sig. diff in Dim.1 (p-value=0.00418), Dim.2 (p-value=0.0298), Dim.2 (p=0.0475) among cultivars.
## These results strongly suggest that the grape cultivars can be differentiated based on their positions along these principal components, demonstrating distinct variability in their measured sensory attributes.