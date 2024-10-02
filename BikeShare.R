library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(poissonreg)
library(glmnet)

#Read in Data and Clean
biketrain <- vroom("train.csv") #read data
biketrain$weather[biketrain$weather == 4] <- 3
biketrain$weather <- as.factor(biketrain$weather)
biketrain$season <- as.factor(biketrain$season)
biketrain$workingday <- as.factor(biketrain$workingday)

biketest <- vroom("test.csv")
biketest$weather[biketest$weather == 4] <- 3
biketest$weather <- as.factor(biketest$weather)
biketest$season <- as.factor(biketest$season)
biketest$workingday <- as.factor(biketest$workingday)

biketrain <- biketrain %>% select(count, everything()) %>%
  select(-registered, -casual)

biketrain <- biketrain %>%
  mutate(
    hour = hour(datetime),
    month = month(datetime, label = TRUE, abbr = TRUE)  
  )  

biketest <- biketest %>%
  mutate(
    hour = hour(datetime),
    month = month(datetime, label = TRUE, abbr = TRUE)  
  ) 

biketrain$month <- as.factor(biketrain$month)
biketrain$hour <- as.factor(biketrain$hour)

biketest$hour <- as.factor(biketest$hour)
biketest$month <- as.factor(biketest$month)



#EDA
plot_bar(biketrain)
plot_histogram(biketrain)
plot_correlation(biketrain)

weatherplot <- ggplot(data = biketrain, mapping= aes(x = weather)) +
  geom_bar(aes(fill = weather)) +
  ggtitle("Bar Plot of Weather") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "none")

humidityplot <- ggplot(data = biketrain, mapping= aes(x = humidity, y = count)) +
  geom_point() +
  geom_smooth(se=FALSE) + 
  ggtitle("Scatter Plot of Humidity") +
  theme(plot.title = element_text(hjust = 0.5))

tempplot <- ggplot(data = biketrain, mapping = aes(x=temp)) +
  geom_histogram(binwidth = 3, fill = 'turquoise', color = 'black') +
  ggtitle("Distribution of Temperature") +
  theme(plot.title = element_text(hjust = 0.5))

workingplot <- ggplot(data = biketrain, mapping = aes(x = workingday)) +
  geom_bar(aes(fill = workingday)) +
  ggtitle("Bar Plot of Working Day") +
  theme(plot.title = element_text(hjust = 0.5), 
        legend.position = "none")


(weatherplot + humidityplot) / (tempplot + workingplot)

#Linear Model
my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>%
  fit(formula=log(count)~., data=biketrain)

bike_predictions <- exp(predict(my_linear_model,
                                new_data=biketest))

#format for kaggle
kaggle_submission <- bike_predictions %>%
  bind_cols(., biketest) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

#Poisson Model
my_poisson_model <- poisson_reg() %>% 
  set_engine("glm") %>% 
  set_mode("regression") %>%
  fit(formula=count~., data=biketrain)

pois_predictions <- predict(my_poisson_model,
                            new_data=biketest) 

pois_kaggle_submission <- pois_predictions %>%
  bind_cols(., biketest) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(datetime=as.character(format(datetime))) 

vroom_write(x=pois_kaggle_submission, file="./PoissonPreds.csv",delim=",")



