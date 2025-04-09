# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)

# Load and prepare data
sensory <- read_csv("final_ordered_wine_data_rf.csv")

# Set factors
# sensory$Grape_Cultivar <- as.factor(sensory$Grape_Cultivar)

# Define features
features <- sensory %>%
  select(
    bottlePrice,
    closureType,
    winepH,
    Red_Fruit_Taste_avg,
    Sweet_Taste_avg,
    Muscadine_Aroma_Intensity_avg,
    Jammy_Taste_avg,
    Jammy_Aroma_avg,
    Sourness_Intensity_avg,
    Dry_Taste_avg,
    Bitter_Taste_avg,
    Herbaceous_Taste_avg,
    Color_Intensity_avg,
    Sweetness_Intensity_avg,
    Muscadine_Taste_avg,
    Muscadine_Flavor_Intensity_avg,
    Oxidized_Aroma_avg,
    Green_Color_avg,
    Citrus_Aroma_avg,
    ABV,
    wineTA,
    wineAbs420,
    wineAbs520,
    wineAbs620,
    colorIntensity,
    tintHue,
    Grape_Cultivar
  )

# Replace NA values with 0
features[is.na(features)] <- 0

# Create the target variables: Overall_Quality_Liking_avg, Taste_Liking_avg, Aroma_Liking_avg
targets <- sensory %>%
  select(Overall_Quality_Liking_avg, Taste_Liking_avg, Aroma_Liking_avg)

# Function to train and evaluate model for a given target variable
train_evaluate_model <- function(features, target, target_name) {
  # Combine features and the target into a single data frame
  data <- cbind(features, target)
  
  # Define the control function for k-fold cross-validation
  set.seed(123)  # For reproducibility
  train_control <- trainControl(method = "cv", number = 10)
  
  # Train the Random Forest model using k-fold cross-validation
  set.seed(123)
  rf_model_cv <- train(
    target ~ .,
    data = data,
    method = "rf",
    trControl = train_control,
    tuneLength = 3  # Number of different mtry values to consider
  )
  
  # Print the results of cross-validation
  print(paste("Results for target:", target_name))
  print(rf_model_cv)
  
  # Feature importance from the cross-validation model
  importance <- varImp(rf_model_cv, scale = FALSE)
  print(importance)
}

# Train and evaluate models for each target variable
train_evaluate_model(features, sensory$Overall_Quality_Liking_avg, "Overall_Quality_Liking_avg")
train_evaluate_model(features, sensory$Taste_Liking_avg, "Taste_Liking_avg")
train_evaluate_model(features, sensory$Aroma_Liking_avg, "Aroma_Liking_avg")

# Taste liking appears to be the best modeled from this dataset.
# It will be interesting to see how adding in the metabolomics data affects the results.
