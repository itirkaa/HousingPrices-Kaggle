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

file_path = 'train.csv'
data = pd.read_csv(file_path)
# dropna drops missing values (think of na as "not available")
#data.dropna(axis=0)

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[feature_names]
y = data.SalePrice

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
preds = forest_model.predict(X_val)
print(mean_absolute_error(y_val, preds))

final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)
test_data = pd.read_csv('test.csv')
X_test = test_data[feature_names]

test_pred = forest_model.predict(X_test)


output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_pred})
output.to_csv('submission.csv', index=False)
