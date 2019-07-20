# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 04:34:43 2019
@author: Aakriti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv', index_col='Id')
test_data = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target
data.dropna(axis=0, subset=['SalePrice'], inplace=True) # removes rows with null SalePrice
y = data.SalePrice # assigns salePrice to y
X = data.drop(['SalePrice'], axis=1) # Removes SalePrice column from data

# Train test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

# "Cardinality" number of unique values in a column
# Categorical values with relatively low cardinality 
categorical_cols = [cname for cname in X_train.columns if X[cname].nunique() < 10 and X[cname].dtype == "object"]

# Numerical columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
all_cols = categorical_cols + numerical_cols
X_train = X_train[all_cols].copy()
X_valid = X_valid[all_cols].copy()
X_test = test_data[all_cols].copy()

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Step 1 Preprocessing / Imputing Numerical Missing Data
numerical_transformer = SimpleImputer(strategy='median') # strategy='constant' replaces missing data with constant fill value, which by default is 0

# Preprocessing of categorical data
categorical_transformer = Pipeline(
        steps=[
                ('imputer', SimpleImputer(strategy='constant')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
        transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)])

# Define Model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundel preprocessing and modeling code in a Pipeline
clf = Pipeline(steps=[('preprocessing', preprocessor),
                      ('model', model)])

clf.fit(X_train, y_train)

preds = clf.predict(X_valid)

print('MAE: ', mean_absolute_error(y_valid, preds))

pred_test = clf.predict(X_test)

output = pd.DataFrame({
        'Id': X_test.index,
        'SalePrice': pred_test})

output.to_csv('submission.csv', index=False)  
  