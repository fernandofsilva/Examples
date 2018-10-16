#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""

import numpy as np
import matplotlib.pyplot as plt

path = "/home/esssfff/Documents/Github/Examples/Datasets/"

points = np.genfromtxt(path+'points.csv', delimiter=',')

xs = points[:,0]
ys = points[:,1]

plt.scatter(xs, ys)
plt.show()

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Load New Points
new_points = points = np.genfromtxt(path+'new_points.csv', delimiter=',')

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

del centroids, centroids_x, centroids_y, labels, new_points, points, xs, ys

samples = np.genfromtxt(path+'seeds_dataset.txt', delimiter='\t')

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Load pandas
import pandas as pd

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

x = ["Kama wheat", "Rosa wheat", "Canadian wheat"]
varieties = [item for item in x for i in range(70)]

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

del ct, df, inertias, labels, k, samples, varieties, x

samples = np.genfromtxt(path+'fish.csv', delimiter=',')
samples = samples[:,1:]

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

species = pd.read_csv(path+"fish.csv", header=None, usecols=[0])[0].tolist()

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels, 'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)

del ct, df, labels, samples, species

movements = np.genfromtxt(path+'company-stock-movements-2010-2015-incl.csv', 
    delimiter=',')
movements = movements[1:,1:]

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

companies = pd.read_csv(path+'company-stock-movements-2010-2015-incl.csv', 
    usecols=['Unnamed: 0'])['Unnamed: 0'].tolist()

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))

del companies, df, labels, movements, path