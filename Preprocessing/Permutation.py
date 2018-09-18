#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:56:25 2018

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
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]

X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
display(eli5.show_weights(perm, feature_names = val_X.columns.tolist()))

