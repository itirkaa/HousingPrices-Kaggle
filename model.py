# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 04:34:43 2019

@author: Aakriti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('train.csv', index_col='Id')
test_data = pd.read_csv('test.csv', index_col='Id')
# dropna drops missing values (think of na as "not available")
#data.dropna(axis=0)
# Remove rows with missing target
data.dropna(axis=0, subset=['SalePrice'], inplace=True) # removes rows with null SalePrice
y = data.SalePrice # assigns salePrice to y
data.drop(['SalePrice'], axis=1, inplace=True) # Removes SalePrice column from data

# Using only numerical predictors
X = data.select_dtypes(exclude=['object'])
X_test = test_data.select_dtypes(exclude=['object'])
"""
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[feature_names]
y = data.SalePrice
"""

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)


# Preliminary investigation
# X_train.shape # shape of training data
missing_val_count = X_train.isnull().sum() # sum of missing values in each column
missing_val_count[missing_val_count > 0] # gives list of columns with missing values greater than 0
"""
result: 
LotFrontage    200
MasVnrArea       7
GarageYrBlt     54
"""

# Funtion for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Approach 1
# Droping missing "COLUMNS" with missing terms
cols_with_missing_val = [col for col in X_train.columns if X_train[col].isnull().any()] # makes list of columns with any null value
# Removing columns from train and validation sets
reduced_X_train = X_train.drop(cols_with_missing_val, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing_val, axis=1)

print("MAE (After dropping columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
"""
result: 
MAE (After dropping columns with missing values):
16926.217502283107
"""

# Approach 2
from sklearn.impute import SimpleImputer
# Imputation: replacing missing values with mean value
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train)) # creates a new dataframe with new imputed values
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
# NOTE: new dataframes have lost column names/headings
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (After imputations):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
"""
result:
MAE (After imputations):
16766.58257534247
"""

# final Approach
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

forest_model = RandomForestRegressor(n_estimators=100, random_state=1)
forest_model.fit(final_X_train, y_train)

preds_valid = forest_model.predict(final_X_valid)
print(mean_absolute_error(y_valid, preds_valid))

# Preprocess test data
final_X_test = pd.DataFrame(imputer.transform(X_test))
preds_test = forest_model.predict(final_X_test)

output = pd.DataFrame({'Id': test_data.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
