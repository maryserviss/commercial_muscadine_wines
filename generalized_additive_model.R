# Load required libraries
library(mgcv)
library(Metrics)
library(dplyr)
library(caret)
library(hydroGOF)

# Set the working directory
setwd("C:/Users/marys/OneDrive/Mary's HP Files/UF/2024-25/Gu's Lab/Wine Modeling/Commercial Wines/Data")

# Read the dataset
data <- read.csv('final_ordered_wine_data_rf.csv')

# Define essential features given the small dataset context
features <- data %>%
  mutate(across(where(is.character), as.factor)) %>%
  select( 
    winepH, 
    Red_Fruit_Taste_avg, 
    Sweet_Taste_avg, 
    Muscadine_Aroma_Intensity_avg, 
    Jammy_Aroma_avg, 
    Sourness_Intensity_avg, 
    Dry_Taste_avg, 
    Bitter_Taste_avg, 
    Sweetness_Intensity_avg, 
    Muscadine_Flavor_Intensity_avg, 
    Grape_Cultivar
  ) %>%
  replace(is.na(.), 0) %>%
  mutate(across(where(is.factor), as.numeric))

# Define the target variable: Taste_Liking_avg
target <- data %>% select(Taste_Liking_avg)

# Combine features and target into one data frame
data_combined <- bind_cols(features, target)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data_combined$Taste_Liking_avg, p = 0.8, list = FALSE, times = 1)
train_data <- data_combined[trainIndex, ]
test_data <- data_combined[-trainIndex, ]

# Simplified formula with interaction terms
simplified_formula <- as.formula(
  "Taste_Liking_avg ~ winepH + Red_Fruit_Taste_avg * Muscadine_Flavor_Intensity_avg * Jammy_Aroma_avg +
   s(Jammy_Aroma_avg, k = 5) + s(Sourness_Intensity_avg, k = 5) + Dry_Taste_avg * Bitter_Taste_avg *
   Sweetness_Intensity_avg * Sweet_Taste_avg + Grape_Cultivar"
)

# LOO Cross-Validation and Metrics Calculation Function
cv_gam_loo <- function(data, formula) {
  n <- nrow(data)
  predictions <- numeric(n)
  actuals <- data$Taste_Liking_avg
  errors <- numeric(n)
  
  for (i in 1:n) {
    train_fold <- data[-i, ]
    test_fold <- data[i, ]
    
    gam_model <- tryCatch(
      {
        gam(formula, data = train_fold, method = "REML", select = TRUE)
      },
      error = function(e) {
        message(paste("Error in observation:", i))
        message(e)
        return(NULL)
      },
      warning = function(w) {
        message(paste("Warning in observation:", i))
        message(w)
        NULL
      }
    )
    
    if (is.null(gam_model)) {
      errors[i] <- NA
      next
    }
    
    prediction <- predict(gam_model, newdata = test_fold)
    predictions[i] <- prediction
    errors[i] <- (prediction - test_fold$Taste_Liking_avg)^2
  }
  
  mse <- mean(errors, na.rm = TRUE)
  rmse <- sqrt(mse)
  nse <- NSE(predictions, actuals)
  kge <- KGE(predictions, actuals)
  
  list(MSE = mse, RMSE = rmse, NSE = nse, KGE = kge)
}

# Perform LOO CV using the simplified formula and calculate metrics
cv_metrics <- cv_gam_loo(data_combined, simplified_formula)

# Print the CV metrics
print(paste("Cross-validated MSE:", cv_metrics$MSE))
print(paste("Cross-validated RMSE:", cv_metrics$RMSE))
print(paste("Cross-validated NSE:", cv_metrics$NSE))
print(paste("Cross-validated KGE:", cv_metrics$KGE))

# Fit the final model using the simplified formula
final_model <- gam(simplified_formula, data = data_combined, method = "REML", select = TRUE)

# Summarize the model to inspect the coefficient estimates and p-values
summary(final_model)

# Plot the residuals to check for any noticeable patterns
par(mfrow = c(2, 2))
plot(final_model)

# Predict the fitted values
fitted_values <- predict(final_model, newdata = data_combined)

# Plot actual vs fitted values
plot(data_combined$Taste_Liking_avg, fitted_values, main = "Actual vs Fitted Values",
     xlab = "Actual Taste Liking Avg", ylab = "Fitted Taste Liking Avg")
abline(0, 1, col = "red")

# Plot residuals
residuals <- data_combined$Taste_Liking_avg - fitted_values
plot(fitted_values, residuals, main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")