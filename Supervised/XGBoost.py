#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""
# Import pandas
import pandas as pd

# Define path to load the files
path = "/home/esssfff/Documents/Git/Examples/Datasets/"

cols = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", 
"bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", 
"appet", "pe", "ane", "class"]

data = pd.read_csv(path+"chronic_kidney_disease.csv", header=None, 
    names=cols, na_values="?")

data["class"] = data["class"].replace(["ckd", "nockd"], [1, 0])

data = data.dropna()

# Import xgboost
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

# Create arrays for the features and the target: X, y
X, y = data.drop("class", axis=1), data["class"]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, 
    random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, 
    seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))