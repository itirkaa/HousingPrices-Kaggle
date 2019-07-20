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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv('train.csv', index_col='Id')
test_data = pd.read_csv('test.csv', index_col='Id')

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

output = pd.DataFrame({'Id': test_data.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

"""
# Approach 1 : dealing with categorical values
# Using only numerical predictors
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
"""
"""
result:
    17591.619006849316
"""
"""
# Approach 2: Label encoding

print("Unique values in 'Conditions2' in training data:", X_train['Condition2'].unique()) # Prints array of unique values in training data
print("Unique values in 'Conditions2' in validation data:", X_test['Condition2'].unique())
"""
"""
On comparing the two arrays wwe notice that some values present in validation data are not present in training data. This implies label encoder never gets to fit in the values missing in training set. This gives an error
Thues it is necessary to make sure that such columns are dealt with or dropped from the training data.
"""
"""
# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype=='object'] # gives list of object type column headings
"""
"""
# Columns that can be easily be label encoded
good_label_cols = [col for col in object_cols if set(X_train[col])==set(X_valid[col])] # gives list of columns which have all the same unique values from valid and training set. Notice only few columns get selected
bad_label_cols = list(set(object_cols)-set(good_label_cols))
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)
label_encoder = LabelEncoder()
for col in set(good_label_cols):
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
print("MAE from Approach 2.1 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
"""
"""
MAE from Approach 2.1 (Label Encoding):
17465.14756849315
"""
"""
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))] # gives list of all the columns where set of unique values from validation set is subset of training set. Notice it has more number of columns in list now

# Problematic columns that will be dropped
bad_label_cols = list(set(object_cols)-set(good_label_cols))
# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in set(good_label_cols):
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

print("MAE from Approach 2.2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
"""
"""
MAE from Approach 2.2 (Label Encoding):
17139.817043378993
"""

# Approach 3: One Hot Encoder
"""
# Checking carinality of a column ie unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols)) # counts the number of unique enteries of each column
d = dict(zip(object_cols, object_nunique))
sorted(d.items(), key=lambda x:x[1])

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10] # List of columns with unique values less than 10 will be one hot encoded

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols])) # low cardinality object columns get one hot encoded. Note it contains only categorical data and has no headings after encoding
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# Reapplying categorial columns headings/index
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# column without any categorical columns/ columns with numerical columns only
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Concatinating numerical features with one hot encoded ones
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
""""""
MAE from Approach 3 (One-Hot Encoding):
17178.240787671235
"""
"""
# Preliminary investigation
# X_train.shape # shape of training data
missing_val_count = X_train.isnull().sum() # sum of missing values in each column
missing_val_count[missing_val_count > 0] # gives list of columns with missing values greater than 0
""""""
result: 
LotFrontage    200
MasVnrArea       7
GarageYrBlt     54
""""""

# Approach 1
# Droping missing "COLUMNS" with missing terms
cols_with_missing_val = [col for col in X_train.columns if X_train[col].isnull().any()] # makes list of columns with any null value
# Removing columns from train and validation sets
reduced_X_train = X_train.drop(cols_with_missing_val, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing_val, axis=1)

print("MAE (After dropping columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
""""""
result: 
MAE (After dropping columns with missing values):
16926.217502283107
""""""

# Approach 2

# Imputation: replacing missing values with mean value
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train)) # creates a new dataframe with new imputed values
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
# NOTE: new dataframes have lost column names/headings
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (After imputations):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
""""""
result:
MAE (After imputations):
16766.58257534247
""""""

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
"""