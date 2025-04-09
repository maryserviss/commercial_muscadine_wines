# Clearing the workspace
rm(list=ls())

# Setting the working directory
setwd("C:/Users/marys/OneDrive/Mary's HP Files/UF/2024-25/Gu's Lab/Wine Modeling/Commercial Wines/Data")

# Loading necessary libraries
library(tidyverse)
library(car)
library(dplyr)
library(Metrics)
library(caret)
library(hydroGOF)
library(MASS)

# Loading the dataset
musc <- read.csv("final_ordered_wine_data_regression_no_wineid.csv")

# # Verifying the data structure and contents
# summary(musc)
# str(musc)

# Fitting a simplified model to ensure the code works 
simplified_model <- lm(Overall_Quality_Liking_avg ~ bottlePrice + ABV + winepH + wineTA +
                         Grape_Cultivar, Brown_Color_avg + Muscadine_Fruity_Aroma_avg +
                         Red_Fruit_Aroma_avg + Apple_Pear_Aroma_avg + Bubblegum_Aroma_avg +
                         Chemical_Aroma_avg + Jammy_Aroma_avg + Floral_Aroma_avg +
                         Herbaceous_Aroma_avg + Green_Grassy_Aroma_avg + Petroleum_Aroma_avg +
                         Sulfidic_Aroma_avg + Sulfite_Aroma_avg + Acetic_Aroma_avg +
                         Oxidized_Aroma_avg + Foxy_Aroma_avg + Muscadine_Taste_avg +
                         Red_Fruit_Taste_avg + + Color_Liking_avg + Aroma_Liking_avg +
                         Taste_Liking_avg, Sweetness_Intensity_avg + Bitterness_Intensity_avg +
                         Sourness_Intensity_avg + Astringency_Intensity_avg, data = musc)
summary(simplified_model)

# Check and handle factors with only one level
single_level_factors <- sapply(musc, function(x) is.factor(x) && length(unique(x)) < 2)
print(single_level_factors)

# Removing single level factors from the dataset and formula
filtered_predictors <- names(musc)[!single_level_factors]
filtered_predictors <- setdiff(filtered_predictors, "Overall_Quality_Liking_avg")

# Fitting an initial full model to check for multicollinearity
initial_full_model <- lm(as.formula(paste("Overall_Quality_Liking_avg ~", paste(filtered_predictors, collapse=" + "))), data = musc)

# Checking for aliased coefficients
aliased_coeffs <- alias(initial_full_model)$Complete
print(aliased_coeffs)

# Removing aliased predictors
aliased_terms <- rownames(aliased_coeffs)
filtered_predictors <- setdiff(filtered_predictors, aliased_terms)

# Creating a formula for the full model without aliased predictors
formula <- as.formula(paste("Overall_Quality_Liking_avg ~", paste(filtered_predictors, collapse=" + ")))

# Fitting the initial full model without aliased predictors
full_model <- lm(formula, data = musc)
summary(full_model)

# Checking VIF values to identify remaining multicollinearity
vif_values <- vif(full_model)
print(vif_values)

# Removing predictors with high VIF values (threshold: 10)
filtered_predictors_vif <- names(vif_values[vif_values < 10])
formula_vif <- as.formula(paste("Overall_Quality_Liking_avg ~", paste(filtered_predictors_vif, collapse=" + ")))

# Fitting the full model again with filtered predictors
full_model_vif <- lm(formula_vif, data = musc)
summary(full_model_vif)

# Proceeding with stepwise selection from full model
stepwise_model <- step(full_model_vif, direction = "both")
summary(stepwise_model)

# Extracting the selected model's formula
stepwise_formula <- formula(stepwise_model)

# Applying 10-fold cross-validation
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)
train_model <- train(stepwise_formula, data = musc, method = "lm", trControl = ctrl)
print(train_model)

# Calculating RMSE from the cross-validation
rmse_value <- sqrt(mean(residuals(stepwise_model)^2))
print(paste("RMSE:", rmse_value))

# Calculating NSE and KGE
observed <- musc$Overall_Quality_Liking_avg
predicted <- predict(stepwise_model, musc)
nse_value <- NSE(predicted, observed)
kge_value <- KGE(predicted, observed)
print(paste("NSE:", nse_value))
print(paste("KGE:", kge_value))

# Calculating accuracy from the cross-validation
accuracy_value <- train_model$results$Rsquared
print(paste("Accuracy:", accuracy_value))

# Generating QQ plot
qqnorm(residuals(stepwise_model), main = "QQ Plot of Residuals")
qqline(residuals(stepwise_model), col = "red")

# Plotting histogram of residuals
hist(residuals(stepwise_model), main = "Histogram of Residuals", xlab = "Residuals", col = "blue", breaks = 30)

# Performing Shapiro-Wilk test for normality
shapiro_test <- shapiro.test(residuals(stepwise_model))
print(shapiro_test)