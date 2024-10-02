library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(GGally)
library(ggplot2)
library(glmnet)
library(ranger)
library(bonsai)
library(lightgbm)

# read in data
train_data <- vroom("train.csv") 
test_data <- vroom("test.csv")

train_data <- train_data %>%
  select(-registered, -casual) %>%
  mutate(count = log(count))

# write recipe
bike_recipe <- recipe(count ~ ., data = train_data) %>%
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = "year") %>%
  step_mutate(
    weather = if_else(weather == 4, 3, weather),
    weather = as.factor(weather),
    season = as.factor(season),
    workingday = as.factor(workingday),
    holiday = as.factor(holiday),
    datetime_year = as.factor(datetime_year)) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


# define model
bart_mod <- bart(mode = "regression", trees=100) %>%
  set_engine("dbarts")

# create workflow
bart_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(bart_mod) %>%
  fit(data=train_data)

bart_preds <- predict(bart_wf, new_data=test_data)

kaggle_submission <- bart_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./BartPreds.csv", delim=",")

