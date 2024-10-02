library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(poissonreg)
library(glmnet)

# Read in Data and Clean
biketrain <- vroom("train.csv")
biketest <- vroom("test.csv")

# Clean biketrain
biketrain$weather[biketrain$weather == 4] <- 3
biketrain$weather <- as.factor(biketrain$weather)
biketrain$season <- as.factor(biketrain$season)
biketrain$workingday <- as.factor(biketrain$workingday)

# Clean biketest
biketest$weather[biketest$weather == 4] <- 3
biketest$weather <- as.factor(biketest$weather)
biketest$season <- as.factor(biketest$season)
biketest$workingday <- as.factor(biketest$workingday)

# Extract hour and month from datetime
biketrain <- biketrain %>%
  mutate(
    hour = hour(datetime),
    month = month(datetime, label = TRUE, abbr = TRUE)
  ) %>%
  select(-datetime, -registered, -casual)  # Remove datetime and other columns

biketest <- biketest %>%
  mutate(
    hour = hour(datetime),
    month = month(datetime, label = TRUE, abbr = TRUE)
  ) 

# Convert to factors
biketrain$month <- as.factor(biketrain$month)
biketrain$hour <- as.factor(biketrain$hour)
biketrain$count <- log(biketrain$count)

biketest$hour <- as.factor(biketest$hour)
biketest$month <- as.factor(biketest$month)

# Penalized regression recipe
bike_recipe <- recipe(count ~ ., data = biketrain) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Prep the recipe
prepped_recipe <- prep(bike_recipe)

# Model specification
preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# Create and fit the workflow
preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) 
## Split data for CV
folds <- vfold_cv(biketrain, v = 5, repeats=1)

CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae, rsq))

## Plot Results (example)
collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best(metric = "rmse")

final_wf <-
preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=biketrain)

# Make predictions
penalized_predictions <- predict(final_wf, new_data = biketest)

# Prepare the Kaggle submission
penalized_kaggle_submission <- penalized_predictions %>%
  bind_cols(biketest) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(datetime = as.character(format(datetime))) %>%
  mutate(count = exp(count))

# Write to CSV
vroom_write(x = penalized_kaggle_submission, file = "./PenalizedPreds.csv", delim = ",")

