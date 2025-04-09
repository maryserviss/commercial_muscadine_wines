# Load required libraries
library(mgcv)
library(keras)
library(caret)
library(Metrics)
library(tidyverse)
library(randomForest)

# Set working directory
setwd("C:/Users/marys/OneDrive/Mary's HP Files/UF/2024-25/Gu's Lab/Wine Modeling/Commercial Wines/Data")

# Read the dataset
data <- read.csv('final_ordered_wine_data_rf.csv')

# Define features
features <- data %>%
  mutate(across(where(is.character), as.factor)) %>%
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
  ) %>%
  replace(is.na(.), 0) %>%
  mutate(across(where(is.factor), as.numeric))

# Define the target variable: Taste_Liking_avg
target <- data %>%
  select(Taste_Liking_avg)

# Combine features and target into one data frame
data_2 <- cbind(features, target)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data_2$Taste_Liking_avg, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- data_2[ trainIndex,]
test_data  <- data_2[-trainIndex,]

# Train a Bagged CART model using caret with better control
set.seed(123)
train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  savePredictions = "final"
)

bagged_cart_model <- train(
  Taste_Liking_avg ~ ., 
  data = train_data, 
  method = "treebag",
  trControl = train_control,
  na.action = na.omit
)

# Print model details
print(bagged_cart_model)

# Predict on the test set
predictions_bagged <- predict(bagged_cart_model, newdata = test_data)

# Compute evaluation metrics for Bagged CART model
mse_bagged <- mse(test_data$Taste_Liking_avg, predictions_bagged)
rmse_bagged <- rmse(test_data$Taste_Liking_avg, predictions_bagged)
nse_bagged <- NSE(predictions_bagged, test_data$Taste_Liking_avg)
kge_bagged <- KGE(predictions_bagged, test_data$Taste_Liking_avg)

# Print evaluation metrics for Bagged CART model
print(paste("Bagged CART - MSE:", mse_bagged))
print(paste("Bagged CART - RMSE:", rmse_bagged))
print(paste("Bagged CART - NSE:", nse_bagged))
print(paste("Bagged CART - KGE:", kge_bagged))

# Train a Random Forest model using the randomForest package
set.seed(123)
rf_model <- randomForest(Taste_Liking_avg ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(train_data)-1))

# Print model details
print(rf_model)

# Predict on the test set
predictions_rf <- predict(rf_model, newdata = test_data)

# Compute evaluation metrics for Random Forest model
mse_rf <- mse(test_data$Taste_Liking_avg, predictions_rf)
rmse_rf <- rmse(test_data$Taste_Liking_avg, predictions_rf)
nse_rf <- NSE(predictions_rf, test_data$Taste_Liking_avg)
kge_rf <- KGE(predictions_rf, test_data$Taste_Liking_avg)

# Print evaluation metrics for Random Forest model
print(paste("Random Forest - MSE:", mse_rf))
print(paste("Random Forest - RMSE:", rmse_rf))
print(paste("Random Forest - NSE:", nse_rf))
print(paste("Random Forest - KGE:", kge_rf))

## rf outperformed decision trees bagged cart in every way