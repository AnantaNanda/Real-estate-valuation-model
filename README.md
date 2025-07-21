# 🏡 Housing Price Prediction

Predicting the price of a property is a critical task for real estate firms to maintain consistent pricing strategies, attract buyers, and evaluate development opportunities. This project implements multiple machine learning models to estimate property prices based on housing data from Melbourne, Australia.

---

## 📁 Project Structure
This project was executed using `R`, with data preprocessing, feature engineering, and model building pipelines structured using the `tidymodels` framework.

---

## 📊 Dataset Description

### 📌 Input Files:
- `housing_train.csv`: Training data with all features including the target variable `Price`.
- `housing_test.csv`: Test data with all features except `Price` (to be predicted).

### 🔑 Key Features Used:
- Location: `Suburb`, `Postcode`, `CouncilArea`
- Property Details: `Rooms`, `Type`, `Method`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`
- Seller/Agent Info: `SellerG`

---

## ⚙️ Workflow Summary

### 🔧 Preprocessing
- Missing value treatment (mean/median imputation)
- Feature encoding:
  - Dummy variable creation for categorical features (`Suburb`, `SellerG`, etc.)
  - Rare level grouping using `step_other()`
- Feature selection using VIF and stepwise regression

### 📈 Models Used
1. **Linear Regression**
   - Performed VIF analysis for multicollinearity
   - Applied stepwise feature elimination
   - Final RMSE-based scoring

2. **Decision Tree**
   - Tuned with grid search (`tree_depth`, `min_n`, `cost_complexity`)
   - Cross-validation with 5-fold CV
   - Visualized tree using `rpart.plot` and feature importance with `vip`

3. **Random Forest**
   - Tuned with `mtry`, `trees`, and `min_n`
   - Feature importance via permutation
   - RMSE used for performance evaluation

4. **XGBoost**
   - Hyperparameter tuning with Latin Hypercube sampling
   - Finalized with best RMSE
   - Feature importance visualization using `vip`

---

## 🔍 Model Evaluation
- Evaluation Metric: **Root Mean Squared Error (RMSE)**
- Internal scoring system: `score = 212467 / RMSE`
- Feature importance plots used to interpret key drivers of price

---

## 📂 Output
The final predicted test values are saved as:

- `Ananta_Nanda_P1_part2.csv` — Linear Regression output
- `Ananta_Nanda_P5(dt)_part2.csv` — Decision Tree output
- `Ananta_Nanda_P1(rf)_part2.csv` — Random Forest output
- `Ananta_Nanda_P1(xgb)_part2.csv` — XGBoost output

---

## 📦 Libraries Used

| R Libraries |
|-------------|
| tidymodels, ranger, xgboost, DALEXtra, vip |
| dplyr, tidyr, car, pROC, ROCit |
| ggplot2, visdat, rpart.plot, forecast |
| doParallel, future, lubridate, stringr, zoo |

---

## 🚀 How to Run

```r
# Set working directory and load data
setwd("your/project/path")
train1 <- read.csv("housing_train.csv")
test1 <- read.csv("housing_test.csv")

# Run model scripts (e.g., linear regression, tree, RF, XGB)
# Each script saves output .csv with predicted prices
```

---

## 🧠 Insights
- **Land size**, **building area**, and **location attributes** play a significant role in predicting housing prices.
- Ensemble models like **XGBoost** and **Random Forest** provided better predictive accuracy over simpler linear models.
- Extensive preprocessing, proper imputation, and encoding were critical to model stability.

---

## 👨‍💻 Author
**Ananta Basudev Nanda**  
B.Tech – Metallurgical and Materials Engineering  
Indian Institute of Technology (IIT), Patna  
📧 anantananda72@gmail.com  
🔗 [GitHub Profile](https://github.com/AnantaNanda)
