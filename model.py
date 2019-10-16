# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 04:34:43 2019

@author: Aakriti
"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv('train.csv', index_col='Id')
test_data = pd.read_csv('test.csv', index_col='Id')

# defining the file path
file_path = 'train.csv'
data = pd.read_csv(file_path)

# list of selective features to do analysis upon
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[feature_names]
y = data.SalePrice

# spliting the training and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

# fitting the model to the training dataset only
# and finding the error using validation dataset
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
preds = forest_model.predict(X_val)
print(mean_absolute_error(y_val, preds))

# fitting the model to the complete dataset
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)
test_data = pd.read_csv('test.csv')
X_test = test_data[feature_names]


test_pred = forest_model.predict(X_test)

# dropna drops missing values (think of na as "not available")
#data.dropna(axis=0)
# Remove rows with missing target
data.dropna(axis=0, subset=['SalePrice'], inplace=True) # removes rows with null SalePrice
y = data.SalePrice # assigns salePrice to y
data.drop(['SalePrice'], axis=1, inplace=True) # Removes SalePrice column from data

#Dropping categorical columns with missing values
object_cols = [col for col in data.columns if data[col].dtype=='object']
object_cols_with_missing = [col for col in object_cols if data[col].isnull().any() or test_data[col].isnull().any()]
X = data.drop(object_cols_with_missing, axis=1)
X_test = test_data.drop(object_cols_with_missing, axis=1)

"""
# Droping columns with missing values
cols_with_missing_val = [col for col in data.columns if data[col].isnull().any()]
X = data.drop(cols_with_missing_val, axis=1)
X_test = test_data.drop(cols_with_missing_val, axis=1)
"""

# Train test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=1)

# Funtion for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print(mean_absolute_error(y_valid, preds))


num_cols = list(set(data.columns)-set(object_cols))

# Step 1 Imputing Numerical Missing Data
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train[num_cols]))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid[num_cols]))

imputed_X_train.columns = X_train[num_cols].columns
imputed_X_valid.columns = X_valid[num_cols].columns

imputed_X_train.index = X_train[num_cols].index
imputed_X_valid.index = X_valid[num_cols].index

score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)

# Step 2 One Hot Encoder
object_cols = list(set(X_train.columns)-set(num_cols))
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_X_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

OH_X_train.index = X_train.index
OH_X_valid.index = X_valid.index

score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)

final_X_train = pd.concat([imputed_X_train, OH_X_train], axis=1)
final_X_valid = pd.concat([imputed_X_valid, OH_X_valid], axis=1)

score_dataset(final_X_train, final_X_valid, y_train, y_valid)

forest_model = RandomForestRegressor(n_estimators=200, random_state=1)
forest_model.fit(final_X_train, y_train)

preds_valid = forest_model.predict(final_X_valid)
print(mean_absolute_error(y_valid, preds_valid))

# Preprocess test data
imputed_X_test = pd.DataFrame(imputer.transform(X_test[num_cols]))
OH_X_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))
imputed_X_test.index = X_test.index
OH_X_test.index = X_test.index
final_X_test = pd.concat([imputed_X_test, OH_X_test], axis=1)

preds_test = forest_model.predict(final_X_test)


output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_pred})
output = pd.DataFrame({'Id': test_data.index,
                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
