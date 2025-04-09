install.packages("party")
install.packages("partykit")
install.packages("hydroGOF")  # For NSE and KGE
library(party)
library(partykit)
library(dplyr)
library(hydroGOF)

data <- read_csv("final_ordered_wine_data_rf.csv")

# Set seed for reproducibility
set.seed(123)

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

# Convert character variables to factors if necessary
features <- features %>%
  mutate_if(is.character, as.factor)

# Create the target variable: Overall_Quality_Liking_avg
target <- sensory %>%
  select(Taste_Liking_avg)

# Combine features and target into one data frame
data <- cbind(features, target)

# Define the formula
target_variable <- "Taste_Liking_avg"
predictor_variables <- names(features)

# Create the formula
formula <- as.formula(paste(target_variable, "~", paste(predictor_variables, collapse = "+")))

# Check the formula
print(formula)

# Set seed for reproducibility
set.seed(123)

# Train the Conditional Inference Random Forest model
cforest_model <- cforest(
  formula,
  data = data,
  control = ctree_control(),
  ntree = 500, mtry = 3)

# Predict on the training data (or use a separate test set if applicable)
predictions <- predict(cforest_model, OOB = TRUE, type = "response")

# Evaluate the model
mean_squared_error <- mean((data$Taste_Liking_avg - predictions)^2)
root_mean_squared_error <- sqrt(mean_squared_error)
r_squared <- cor(data$Taste_Liking_avg, predictions)^2
nse <- NSE(predictions, data$Taste_Liking_avg)
kge <- KGE(predictions, data$Taste_Liking_avg)

# Print metrics
print(paste("Mean Squared Error: ", mean_squared_error))
print(paste("Root Mean Squared Error: ", root_mean_squared_error))
print(paste("R-squared: ", r_squared))
print(paste("Nash-Sutcliffe Efficiency: ", nse))
print(paste("Kling-Gupta Efficiency: ", kge))

# Evaluate variable importance
varimp <- varimp(cforest_model)
if(length(varimp) == 0) {
  cat("Warning: Variable importance (standard method) returned empty results.\n")
}
print(varimp)

varimp_conditional <- varimp(cforest_model, conditional = TRUE)
if(length(varimp_conditional) == 0) {
  cat("Warning: Variable importance (conditional method) returned empty results.\n")
}
print(varimp_conditional)
