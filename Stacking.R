library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(GGally)
library(ggplot2)
library(glmnet)
library(ranger)
library(stacks)

# read in data
train_data <- vroom("train.csv") 
test_data <- vroom("test.csv")

train_data <- train_data %>%
  select(-registered, -casual) %>%
  mutate(count = log(count))

# write recipe
bike_recipe <- recipe(count ~ ., data = train_data) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(
    weather = if_else(weather == 4, 3, weather),
    weather = as.factor(weather),
    season = as.factor(season),
    workingday = as.factor(workingday),
    holiday = as.factor(holiday)) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

folds <- vfold_cv(test_data, v = 5, repeats=1)

untunedModel <- control_stack_grid() 
tunedModel <- control_stack_resamples()

# Model specification
preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# Create and fit the workflow
preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) 

preg_models <- preg_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq),
          control = untunedModel)

# define model
tree_mod <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=500) %>%
  set_engine("ranger") %>% 
  set_mode("regression")

# create workflow
tree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tree_mod)

# finalize workflow and make predictions
tree_model <- rand_forest(mtry = 10, 
                          min_n = 2,
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

tuning_params <- grid_regular(mtry(range = c(1,10)),
                              min_n(),
                              levels = 5)

tree_models <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_params,
            metrics=metric_set(rmse, mae, rsq))

my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(tree_models)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

stackData <- as.tibble(my_stack)

stack_mod %>% predict(new_data=biketest)

kaggle_submission <- stack_mod %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./StackPreds.csv", delim=",")


