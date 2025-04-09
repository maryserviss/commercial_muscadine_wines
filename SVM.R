install.packages("e1071")
install.packages("kernlab")
install.packages('DALEX')
library(DALEX)
library(e1071)
library(dplyr)
library(hydroGOF)
library(caret)
library(kernlab)

data <- read.csv("final_ordered_wine_data_rf.csv")

# Set seed for reproducibility
set.seed(123)

# Define features and ensure character to factor conversion
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

# Define the target variable: Taste_Liking_avg
target <- data %>%
  select(Taste_Liking_avg)

# Combine features and target into one data frame
data <- cbind(features, target)

# Define the formula
target_variable <- "Taste_Liking_avg"
predictor_variables <- names(features)
formula <- as.formula(paste(target_variable, "~", paste(predictor_variables, collapse = "+")))

# Train the SVM model
svm_model <- svm(formula, data = data, type = 'eps-regression')

# Predict on the training data (or use a separate test set if applicable)
svm_predictions <- predict(svm_model, data)

# Remove potential NA values in predictions
svm_predictions <- na.omit(svm_predictions)

# Calculate performance metrics
mean_squared_error <- mean((data$Taste_Liking_avg - svm_predictions)^2, na.rm = TRUE)
root_mean_squared_error <- sqrt(mean_squared_error)
r_squared <- cor(data$Taste_Liking_avg, svm_predictions, use = "complete.obs")^2
nse <- NSE(svm_predictions, data$Taste_Liking_avg)
kge <- KGE(svm_predictions, data$Taste_Liking_avg)

# Print performance metrics
print(paste("Mean Squared Error: ", mean_squared_error))
print(paste("Root Mean Squared Error: ", root_mean_squared_error))
print(paste("R-squared: ", r_squared))
print(paste("Nash-Sutcliffe Efficiency: ", nse))
print(paste("Kling-Gupta Efficiency: ", kge))

### Cross-Validation
# Define train control for cross-validation
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train the SVM model with cross-validation
svm_cv_model <- train(
  formula, data = data, method = "svmRadial",
  trControl = train_control,
  preProcess = c("center", "scale")  # Optionally preprocess by centering and scaling data
)

# Print cross-validation results
print(svm_cv_model)

# Extract predictions
cv_predictions <- predict(svm_cv_model, data)

# Calculate performance metrics on cross-validated model
mean_squared_error_cv <- mean((data$Taste_Liking_avg - cv_predictions)^2, na.rm = TRUE)
root_mean_squared_error_cv <- sqrt(mean_squared_error_cv)
r_squared_cv <- cor(data$Taste_Liking_avg, cv_predictions, use = "complete.obs")^2
nse_cv <- NSE(cv_predictions, data$Taste_Liking_avg)
kge_cv <- KGE(cv_predictions, data$Taste_Liking_avg)

# Print performance metrics for cross-validated model
print(paste("Cross-validated Mean Squared Error: ", mean_squared_error_cv))
print(paste("Cross-validated Root Mean Squared Error: ", root_mean_squared_error_cv))
print(paste("Cross-validated R-squared: ", r_squared_cv))
print(paste("Cross-validated Nash-Sutcliffe Efficiency: ", nse_cv))
print(paste("Cross-validated Kling-Gupta Efficiency: ", kge_cv))

# Implementing feature importance with permutation
# Create an explainer for the SVM model
explainer_svm <- explain(svm_model, data = features, y = data$Taste_Liking_avg)

# Calculate permutation feature importance
importance_svm <- model_parts(explainer_svm, N = 500)  # N: number of permutations

# Plot the importance
plot(importance_svm)
