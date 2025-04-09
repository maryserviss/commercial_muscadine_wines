install.packages("xgboost")
install.packages("dplyr")
install.packages("magrittr")
install.packages("caret")
library(caret)
library(xgboost)
library(dplyr)
library(hydroGOF)

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
  )

# Replace NAs with zeros
features[is.na(features)] <- 0

# Convert factor variables to numeric for XGBoost
features <- features %>% mutate_if(is.factor, as.numeric)

# Define the target variable: Taste_Liking_avg
target <- data %>%
  select(Taste_Liking_avg)

# Combine features and target into one data frame
data_2 <- cbind(features, target)

# Convert to XGBoost matrix
dtrain <- xgb.DMatrix(data = as.matrix(features), label = target$Taste_Liking_avg)

# Set seed for reproducibility
set.seed(123)

# Define parameters for the XGBoost model
params <- list(
  objective = "reg:squarederror",  # Regression objective
  eval_metric = "rmse"             # Evaluation metric
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100  # Number of boosting rounds
)

# Predict on the training data (or use a separate test set if applicable)
xgb_predictions <- predict(xgb_model, as.matrix(features))

# Remove potential NA values in predictions
xgb_predictions <- na.omit(xgb_predictions)

# Calculate performance metrics
mean_squared_error_xgb <- mean((data_2$Taste_Liking_avg - xgb_predictions)^2, na.rm = TRUE)
root_mean_squared_error_xgb <- sqrt(mean_squared_error_xgb)
r_squared_xgb <- cor(data_2$Taste_Liking_avg, xgb_predictions, use = "complete.obs")^2
nse_xgb <- NSE(xgb_predictions, data_2$Taste_Liking_avg)
kge_xgb <- KGE(xgb_predictions, data_2$Taste_Liking_avg)

# Print performance metrics
print(paste("Mean Squared Error: ", mean_squared_error_xgb))
print(paste("Root Mean Squared Error: ", root_mean_squared_error_xgb))
print(paste("R-squared: ", r_squared_xgb))
print(paste("Nash-Sutcliffe Efficiency: ", nse_xgb))
print(paste("Kling-Gupta Efficiency: ", kge_xgb))

## Cross-Validation
# Install and load caret package
install.packages("caret")
library(caret)

# Define train control for cross-validation
train_control <- trainControl(
  method = "cv", 
  number = 10,  # 10-fold cross-validation
  verboseIter = TRUE  # Print training log
)

# Train the XGBoost model with cross-validation
xgb_cv_model <- train(
  Taste_Liking_avg ~ .,
  data = data_2,
  method = "xgbTree",
  trControl = train_control,
  preProcess = c("center", "scale"),  # Optionally preprocess by centering and scaling data
  tuneLength = 3  # Number of tuning parameters to try
)

# Print cross-validation results
print(xgb_cv_model)

# Extract predictions
cv_predictions <- predict(xgb_cv_model, data_2)

# Calculate performance metrics on cross-validated model
mean_squared_error_cv <- mean((data_2$Taste_Liking_avg - cv_predictions)^2, na.rm = TRUE)
root_mean_squared_error_cv <- sqrt(mean_squared_error_cv)
r_squared_cv <- cor(data_2$Taste_Liking_avg, cv_predictions, use = "complete.obs")^2
nse_cv <- NSE(cv_predictions, data_2$Taste_Liking_avg)
kge_cv <- KGE(cv_predictions, data_2$Taste_Liking_avg)

# Print performance metrics for cross-validated model
print(paste("Cross-validated Mean Squared Error: ", mean_squared_error_cv))
print(paste("Cross-validated Root Mean Squared Error: ", root_mean_squared_error_cv))
print(paste("Cross-validated R-squared: ", r_squared_cv))
print(paste("Cross-validated Nash-Sutcliffe Efficiency: ", nse_cv))
print(paste("Cross-validated Kling-Gupta Efficiency: ", kge_cv))

