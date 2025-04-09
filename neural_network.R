install.packages("keras")
install.packages("neuralnet")
install.packages("caret")
install.packages("Metrics")
install.packages("reticulate")
install.packages("dplyr")
library(dplyr)
library(neuralnet)
library(keras)
library(caret)
library(Metrics)
library(reticulate)

# Use the correct Python virtual environment
use_python("C:/Users/marys/OneDrive/Mary's HP Files/UF/2024-25/Gu's Lab/Wine Modeling/Commercial Wines/Code/r-tensorflow/Scripts/python.exe", required = TRUE)

# Check Python Configuration to make sure everything is detected
py_config()

# Ensure TensorFlow and NumPy can be imported correctly
py_run_string("import numpy")
py_run_string("import tensorflow")
py_run_string("print(tensorflow.__version__)")

setwd("C:/Users/marys/OneDrive/Mary's HP Files/UF/2024-25/Gu's Lab/Wine Modeling/Commercial Wines/Data")
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

# Convert factor variables to numeric
features <- features %>% mutate_if(is.factor, as.numeric)

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

# Train a neural network with neuralnet package
nn_formula <- as.formula(paste('Taste_Liking_avg ~', paste(names(features), collapse = ' + ')))
nn <- neuralnet(nn_formula, data=train_data, hidden=c(5,3), linear.output=TRUE)

# Visualize the neural network
plot(nn)

# Predict using neuralnet model
nn_predictions <- compute(nn, test_data[,1:ncol(test_data)-1])
nn_predictions <- nn_predictions$net.result
nn_mse <- mse(test_data$Taste_Liking_avg, nn_predictions)

cat('neuralnet MSE:', nn_mse, '\n')

# Using keras package to build a neural network model
# Prepare the data for keras
x_train <- as.matrix(train_data[ , -ncol(train_data)])
y_train <- as.matrix(train_data$Taste_Liking_avg)
x_test <- as.matrix(test_data[ , -ncol(test_data)])
y_test <- as.matrix(test_data$Taste_Liking_avg)

# Define the keras model / start here
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1)

# Compile the keras model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
  metrics = list('mean_squared_error')
)

# Fit the keras model
history <- model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

# Plot the training history
plot(history)

# Evaluate the keras model
evaluate(model, x_test, y_test)

# Predict using keras model
keras_predictions <- model %>% predict(x_test)
keras_mse <- mse(y_test, keras_predictions)

cat('keras MSE:', keras_mse, '\n')

