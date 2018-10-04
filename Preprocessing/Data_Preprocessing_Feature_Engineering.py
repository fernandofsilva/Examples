#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fernando Silva
@description: This script contains the preprocessing and feature engineering
also there is a modeling using knn and naives baies to predict the country
where the UFO appear and the type of the UFO
"""

# Loading libries
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

path = '/home/esssfff/Documents/Github/Examples/Datasets/'

# Load the dataset
with open(path+'ufo_sightings_large.csv', 'r') as file:
    ufo = pd.read_csv(file)

del path

# Check how many values are missing in the length_of_time, state, and 
# type columns
print(ufo[['length_of_time', 'state', 'type']].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo = ufo[ufo['length_of_time'].notnull() & 
    ufo['state'].notnull() & 
    ufo['type'].notnull()]

def return_minutes(time_string):

    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")
    
    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(lambda row: return_minutes(row))

# Take a look at the head of both of the columns
print(ufo[['length_of_time', 'minutes']].head())

# Check the variance of the seconds and minutes columns
print(ufo[['seconds','minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])
ufo = ufo.replace([np.inf, -np.inf], np.nan)
ufo = ufo.dropna(how='any')

# Print out the variance of just the seconds_log column
print(ufo['seconds_log'].var())

# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda row: 1 if row=='us' else 0)

# Print the number of unique type values
print(len(ufo['type'].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)
del type_set

# Look at the first 5 rows of the date column
print(ufo['date'].head(5))

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda row: pd.to_datetime(row).month)

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda row: pd.to_datetime(row).year)

# Take a look at the head of all three columns
print(ufo[['date', 'month', 'year']].head(5))

# Take a look at the head of the desc field
print(ufo['desc'].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns this creates
print(desc_tfidf.shape)

# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds', 'seconds_log', 'minutes']].corr())

# Make a list of features to drop
to_drop = ['date', 'seconds', 'minutes', 'desc', 'city', 'country', 'lat', 
           'long', 'recorded', 'state', 'length_of_time']

# Drop those features
ufo = ufo.drop(columns=to_drop)
del to_drop

# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

vocab = {v:k for k,v in vec.vocabulary_.items()}

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]
del vocab

# Modeling with Knn

X = ufo.drop(columns=['country_enc', 'type'])
y = ufo[['country_enc']]

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify=y)

# Defining the model
knn = KNeighborsClassifier()

# Fit knn to the training sets
knn.fit(train_X, train_y.values.ravel())

# Print the score of knn on the test sets
print(knn.score(test_X, test_y))


# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Defining the model
nb = GaussianNB()

# Fit nb to the training sets
nb.fit(train_X, train_y.values.ravel())

# Print the score of nb on the test sets
print(nb.score(test_X, test_y))


