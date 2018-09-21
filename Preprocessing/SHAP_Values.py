#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 11:09:42 2018

@author: esssfff
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path = "/home/esssfff/Documents/Github/Examples/Datasets/"

data = pd.read_csv(path+"FIFA_2018_Statistics.csv")
del path

# Convert from string "Yes"/"No" to binary
y = (data['Man of the Match'] == "Yes")

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# package used to calculate Shap values
import shap

row_to_show = 5

# use 1 row of data here. Could use multiple rows if desired
data_for_prediction = val_X.iloc[row_to_show]  
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

my_model.predict_proba(data_for_prediction_array)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

