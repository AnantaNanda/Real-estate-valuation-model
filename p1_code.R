library(tidymodels)
library(dplyr)
library(car)
library(ROCit)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(visdat)
library(doParallel)
registerDoParallel(cores=4)
library(tidyr)
library(rpart.plot)
library(DALEXtra)
library(ranger)
library(vip)
getwd()
setwd("C:/Users/OMEN/OneDrive/Desktop/T.P/p1")

train1 = read.csv("housing_train.csv",stringsAsFactors=F)
test1 = read.csv("housing_test.csv",stringsAsFactors=F)

glimpse(train1)
nrow(train1)

######## LINEAR MODEL #########

df= train1
delete.na = function(DF, n=0) {
  DF[rowSums(is.na(DF)) <= n,]
}
df_1 = delete.na(df,3)
nrow(df_1)
df_1$Postcode = as.character(df_1$Postcode)
glimpse(df_1)

# Suburb : Create dummies
# Address : Drop it
# rooms :Keep it
# type : : Create dummies 
# method : Create dummies
# seller g : Create dummies
# distance : keep it
# post cod : First convert it into char & then Create dummies
# bedroom2 : keep it
# bathroom : keep it
# car :  keep it
# land size :  keep it
# building area :  keep it
# year built: :  keep it
# council area : Create dummies
# price : target var

## DATA PREPARATION ##

## Sets an outline specifying the dependent variable
dt_pipe = recipe(Price ~ . , data = df_1) %>% 
  # We need to use tidymodels lib to update roles
  update_role(Address,new_role = "drop_vars") %>% 
  # update_role(Postcode,new_role = "to_character") %>% 
  update_role(Suburb,Type,Method,SellerG,Postcode,CouncilArea,new_role = "create_dummies") %>%
  
  # In order to take action as per designed role
  step_rm(has_role("drop_vars")) %>% 
  
  # # TO convert into categorical variable
  # step_mutate_at(has_role("to_character"),fn = as.character()) %>% 
  
  ## to handle missing values(unknown values) in dummies
  step_novel(SellerG,role = NA) %>% 
  step_unknown(has_role("create_dummies"),new_level = "__missing__") %>% 
  
  ## for clubbing data into dummies, built in func step_other is used 
  step_other(has_role("create_dummies"),threshold = 0.02,other="__other__") %>% 
  
  ## to prepare dummy variable
  step_dummy(has_role("create_dummies")) %>% 
  
  step_impute_mean(BuildingArea) %>% 
  #To fill missing values
  step_impute_median(all_numeric(),-all_outcomes())

dt_pipe = prep(dt_pipe)

df_train1 = bake(dt_pipe,new_data = NULL)
df_test1 = bake(dt_pipe,new_data = test1)

View(df_train1)
vis_dat(df_train1) # Has only numeric values
vis_dat(train1) # Has every value(datatype)

## DATA SAMPLING ##
set.seed(3)
s = sample(1:nrow(df_1),0.8 * nrow(df_1))
train_data = df_train1[s,]
test_data = df_train1[-s,]
# nrow(train_data)
glimpse(train_data)


fit_1 = lm(Price ~ . ,data = train_data)
summary(fit_1)
formula(fit_1)

fit_1 = lm(Price ~ .-Type_X__other__-Postcode_X3165-Postcode_X3141-
             Postcode_X3072-Postcode_X__other__-SellerG_Woodards-
             SellerG_Buxton -Postcode_X3121 -SellerG_Ray -SellerG_Barry
             -Postcode_X3032-SellerG_Brad -Postcode_X3020 -SellerG_Biggin
              -Bedroom2-CouncilArea_Moonee.Valley-Suburb_Preston-Method_X__other__
             -Postcode_X3163-SellerG_Fletchers, data = train_data)
summary(fit_1)
# To check n reduce redundancy(multi-collinearity), we use vif function
# Variance Inflation factor
sort(vif(fit_1),decreasing=T)[1:3]
fit_1 = lm(Price ~ .-Type_X__other__-Postcode_X3165-Postcode_X3141-
             Postcode_X3072-Postcode_X__other__-SellerG_Woodards-
             SellerG_Buxton -Postcode_X3121 -SellerG_Ray -SellerG_Barry
           -Postcode_X3032-SellerG_Brad -Postcode_X3020 -SellerG_Biggin
           -CouncilArea_Moonee.Valley-Bedroom2-Suburb_Preston-Method_X__other__
           -Postcode_X3163-SellerG_Fletchers-Suburb_X__other__, data = train_data)

fit_1= stats::step(fit_1)
formula(fit_1)
summary(fit_1)

predicted_data = abs(round(predict(fit_1,newdata = test_data),2))
View(test_data)
View(train1)
head(predicted_data)

errors = test_data$Price - predicted_data

## To determine RMSE :
rmse=errors**2 %>% 
  mean() %>% 
  sqrt()

rmse

score = 212467/rmse
score

fit_final1 = lm(Price ~ .-Type_X__other__-Postcode_X3165-Postcode_X3141-
             Postcode_X3072-Postcode_X__other__-SellerG_Woodards-
             SellerG_Buxton -Postcode_X3121 -SellerG_Ray -SellerG_Barry
           -Postcode_X3032-SellerG_Brad -Postcode_X3020 -SellerG_Biggin
           -CouncilArea_Moonee.Valley-Bedroom2-Suburb_Preston-Method_X__other__
           -Postcode_X3163-SellerG_Fletchers-Suburb_X__other__, data = df_train1)
summary(fit_final1)
sort(vif(fit_final1),decreasing = T)[1:3]
fit_final1=stats::step(fit_final1)

fin_pred1=predict(fit_final1,df_test1)
head(fin_pred1)
# length(fin_pred1)
# nrow(df_test1)
# nrow(test1)


write.csv(fin_pred1,"Ananta_Nanda_P1_part2.csv",row.names = T)


##### DECISION TREE MODEL #####
################
# Parameter tuning for regression
################

tree_model=decision_tree(
  cost_complexity = tune(),
  min_n = tune(),
  tree_depth = tune()
) %>% 
  set_engine("rpart") %>%  #in the back end "rpart" does all the role
  set_mode("regression") # regression prblm it is


## Decision tree argument explanation:
# cost_complexity : complexity parameter. Any split that does not 
# decrease the overall lack of fit by a factor of cp is not attempted.
# The main role of this parameter is to save computing time by pruning off 
# splits that are obviously not worthwhile.(decrease in rmse should be significant)

# tree_depth : Set the maximum depth of any node of the final tree,
# with the root node counted as depth 0. 

# min_n : the minimum number of observations that must exist in a node 
# in order for a split to be attempted.


################
# Grid-search with Cross-validation
################

# K-Fold cross validation(divides the data into 5 parts and tests data on 1 and trains data on 4 parts)
# CROSS-VALIDATION
folds=vfold_cv(df_train1,v=5)

# This is to search the best condition of predicting (GRID-SEARCH)

tree_grid=grid_regular(
  min_n(),
  cost_complexity(),
  tree_depth(),
  levels = 5
)
# Check the extreme values of the 3 hyper-parameters and levels make equal cuts between the two extremes
# It will try 4 combinations from each parameter and make 4*4*4 trees



View(tree_grid)

# ?tree_depth # To see the range of min and max values used in tree_depth
# # tree_depth(range = c(1L, 15L), trans = NULL) -> It means that min value of tree_depth is 1 and max tree_depth is 15

# Different number of combinations to try out. Change levels()
# tree_grid2 = grid_regular(cost_complexity(),
#                          tree_depth(),
#                          min_n(),
#                          levels = c(2,3,4)) # Total combination to be tried = 2*3*4 => 24
# View(tree_grid2)


# Run grid-search with cross-validation parallelly

doParallel::registerDoParallel()

library(future)
plan(multisession, workers = 4)

my_res=tune_grid(
  tree_model,
  Price ~ .,
  resamples = folds ,
  grid = tree_grid,
  metrics = metric_set(yardstick::rmse) # There can be other metrics also like (RMSE or MAE)
)

fold_metrics=collect_metrics(my_res) # To view it in the form of dataframe
View(fold_metrics)

#To view in terms of plot
# autoplot(my_res)+theme_light()

# To see the top % best parameters

k=my_res %>%
  show_best(metric = "rmse")

# k[1,]   Both of them r same
# best_prmtr

best_prmtr=select_best(my_res,metric="rmse")
best_prmtr

#####
# To finally build a model 
#####
final_tree_fit=tree_model %>% 
  finalize_model(best_prmtr) %>% 
  fit(Price ~ . ,data = df_train1)


################
# Feature importance 
################


final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "red", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

## Determines to what extent the dependent variables are important

rpart.plot(final_tree_fit$fit,roundint = F) # cex stands for "character expansion"
final_tree_fit


################
# Model prediction 
################

train_pred=round(predict(final_tree_fit,df_train1),0)
head(train_pred)
test_pred=round(predict(final_tree_fit,df_test1),0)

error_dt = df_train1$Price - train_pred




nrow(error_dt)
nrow(train_pred)
sum(is.na(erros_1))
rmse=sqrt(sum(error_dt^2)/nrow(error_dt))
## To determine RMSE :
rmse=erros_1^2 %>% 
  mean() %>% 
  sqrt()

rmse

score = 212467/rmse
score


head(test_pred)
write.csv(test_pred,"Ananta_Nanda_P5(dt)_part2.csv",row.names = F)

################
################
##### RANDOM FOREST MODEL #####
################
################

rf_model=rand_forest(
  mtry = tune(), #The number of predictors that will be randomly sampled at each split when creating tree models
  trees = tune(), #the number of trees to be considered 
  min_n = tune() # the minimum number of parameters to be considered
) %>% 
  set_mode("regression") %>%
  set_engine("ranger")

vfolds=vfold_cv(df_train1,v=5)

rf_grid=grid_regular(
  mtry(c(5,30)),
  trees(c(10,500)),
  min_n(c(2,10)),
  levels = 5
)

doParallel::registerDoParallel()

library(future)
plan(multisession, workers = 4)

my_res2=tune_grid(
  rf_model,
  Price ~ .,
  resamples = vfolds,
  grid = rf_grid,
  metrics = metric_set(yardstick::rmse)
)


# autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res2)
View(fold_metrics)

k2=my_res2 %>% 
  show_best(metric = "rmse")

k2[1,]

final_rf_fit=rf_model %>% 
  set_engine("ranger", importance = 'permutation') %>% 
  finalize_model(select_best(my_res2,metric="rmse")) %>% 
  fit(Price ~ .,data=df_train1)

# Variable importance 
final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "green", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# Prediction
train_pred1=round(predict(final_rf_fit,new_data = df_train1),0)
head(train_pred1)



test_pred1=round(predict(final_rf_fit,new_data = df_test1),0)
head(test_pred1)

error_rf = df_train1 $ Price - train_pred1


rmse = sqrt(sum(error_rf^ 2)/nrow(error_rf))
rmse


score = 212467/rmse
score
write.csv(test_pred1,"Ananta_Nanda_P1(rf)_part2.csv",row.names = F)


####### XGBOOST MODEL #########

library(tidymodels)
library(doParallel)
library(vip)
library(forecast)
library(fpp)
library(stringr)
library(lubridate)
library(zoo)
library(xgboost)
library(future)
plan(multisession, workers = 4)
# Ensure parallelism
doParallel::registerDoParallel()

# Define XGBoost model with tunable parameters
xgb_spec = boost_tree(
  trees = 300,              # Fixed number of trees (you can increase to ~1000 if needed)
  tree_depth = tune(), 
  min_n = tune(), 
  sample_size = tune(), 
  mtry = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

# Define parameter grid
xgb_grid = grid_latin_hypercube(
  tree_depth(),
  min_n(),
  sample_size = sample_prop(), 
  finalize(mtry(), df_train1),
  size = 30  # Try more if time permits
)

# Cross-validation
set.seed(123)
xgb_folds = vfold_cv(df_train1, v = 5)

# Build workflow
xgb_wf = workflow() %>%
  add_formula(Price ~ .) %>%
  add_model(xgb_spec)

# Tune hyperparameters using grid search
xgb_res = tune_grid(
  xgb_wf,
  resamples = xgb_folds,
  grid = xgb_grid,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# View results
collect_metrics(xgb_res) %>% View()
show_best(xgb_res, metric = "rmse", n = 5)

# Select best model
best_rmse = select_best(xgb_res, metric = "rmse")
best_rmse

# Finalize workflow using best hyperparameters
final_xgb = finalize_workflow(xgb_wf, best_rmse)

# Fit final model on entire training data
final_xgb_fit = final_xgb %>%
  fit(data = df_train1) %>%
  extract_fit_parsnip()

# Variable importance plot
final_xgb_fit %>%
  vip(geom = "col", aesthetics = list(fill = "purple", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# Predictions
train_pred_xgb = round(predict(final_xgb_fit, new_data = df_train1), 0)
test_pred_xgb = round(predict(final_xgb_fit, new_data = df_test1), 0)

# RMSE calculation on training data
error_xgb = df_train1$Price - train_pred_xgb$.pred
rmse_xgb = sqrt(mean(error_xgb^2))
rmse_xgb

# Score
score = 212467 / rmse_xgb
score

# Export test predictions
write.csv(test_pred_xgb, "Ananta_Nanda_P1(xgb)_part2.csv", row.names = F)

