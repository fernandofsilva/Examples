#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np

# Defining default path
path = "/home/esssfff/Documents/Github/Examples/Datasets/"

# Load dataset
df = pd.read_csv(path+"hourly_wages.csv")

# Split between predictors and target
predictors = df.drop("wage_per_hour", axis=1).values
target = df['wage_per_hour'].values

# Specify the model for regression
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Fit the model
model.fit(predictors, target)

del df, n_cols, predictors, target

# Import necessary modules
from keras.utils import to_categorical

# Load dataset
df = pd.read_csv(path+"titanic_all_numeric.csv")

# Split between predictors and target
predictors = df.drop("survived", axis=1).values
target = to_categorical(df['survived'].values)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)

pred_data = pd.read_csv(path+'pred_titanic.csv', header=None)
pred_data[6] = pred_data[6].astype('bool')
pred_data = pred_data.values

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)






