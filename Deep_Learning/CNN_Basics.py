#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""
# Import matplotlib
import matplotlib.pyplot as plt

path = "/home/esssfff/Documents/Github/Examples/Datasets/"

# Load the image
data = plt.imread(path+'bricks.png')

# Display the image
plt.imshow(data)
plt.show()

# Set the red channel in this part of the image to 1
data[:10, :10, 0] = 1

# Set the green channel in this part of the image to 0
data[:10, :10, 1] = 0

# Set the blue channel in this part of the image to 0
data[:10, :10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()

del data

# one-hot encoding to represent images

# Import numpy
import numpy as np

labels = ['shoe', 'shirt', 'shoe', 'shirt', 'dress', 'dress', 'dress']

# The number of image categories
n_categories = 3

# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])

# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))

# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1

del categories, ii, jj, labels, n_categories, ohe_labels

test_labels = np.array([[0., 0., 1.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.],
                        [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 1., 0.]])

predictions = np.array([[0., 0., 1.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.],
                        [0., 0., 1.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

# Calculate the number of correct predictions
number_correct = (test_labels * predictions).sum()
print(number_correct)

# Calculate the proportion of correct predictions
proportion_correct = number_correct / len(test_labels) 
print(proportion_correct)

del number_correct, predictions, proportion_correct, test_labels

# load dataset
data = np.load(path+"fashion.npz")
train_data = data['arr_0'].item()['train_data']
train_labels = data['arr_0'].item()['train_labels']
test_data = data['arr_0'].item()['test_data']
test_labels = data['arr_0'].item()['test_labels']
del data

# Imports components from Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializes a sequential model
model = Sequential()

# First layer
model.add(Dense(10, activation='relu', input_shape=(784,)))

# Second layer
model.add(Dense(10, activation='relu'))

# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Reshape the data to two-dimensional array
train_data = train_data.reshape(50, 784)

# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)


# Reshape test data
test_data = test_data.reshape(10, 784)

# Evaluate the model
model.evaluate(test_data, test_labels)

# One dimensional convolutions
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

# Print conv
print(conv)

del array, conv, ii, kernel

# Convolutional network for image classification

# load dataset
data = np.load(path+"fashion.npz")
train_data = data['arr_0'].item()['train_data']
train_labels = data['arr_0'].item()['train_labels']
test_data = data['arr_0'].item()['test_data']
test_labels = data['arr_0'].item()['test_labels']
del data

# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Initialize the model object
model = Sequential()

# define variables
img_rows = 28
img_cols = 28

# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Flatten the output of the convolutional layer
model.add(Flatten())

# Add an output layer for the 3 categories
model.add(Dense(3, activation='softmax'))

# Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Fit the model on a training set
model.fit(train_data, train_labels, validation_split=0.2, epochs=3, 
    batch_size=10)

# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size=10)


# Add padding to a CNN
# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
    input_shape=(img_rows, img_cols, 1), padding='same'))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Add strides to a convolutional network
# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
    input_shape=(img_rows, img_cols, 1), strides=2))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Creating a deep learning network
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size=2, activation='relu', 
input_shape=(img_rows, img_cols, 1)))

# Add another convolutional layer (5 units)
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Fit the model to training data 
model.fit(train_data, train_labels, validation_split=0.2, 
    epochs=3, batch_size=10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size=10)


# Keras pooling layers

from keras.layers import MaxPool2D

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu', 
                 input_shape=(img_rows, img_cols, 1)))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Fit to training data
model.fit(train_data, train_labels, validation_split=0.2, epochs=3, 
    batch_size=10)

# Evaluate on test data 
model.evaluate(test_data, test_labels, batch_size=10)

# Plot the learning curves

# Train the model and store the training object
training = model.fit(train_data, train_labels, validation_split=0.2, epochs=3, batch_size=10)

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()


# Using stored weights to predict in a test set

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Add another convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Load the weights from file
model.load_weights(path+'weights.hdf5')

# Predict from the first three images in the test data
model.predict(test_data[:3])

# Adding dropout to your network

from keras.layers import Dropout

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Add a dropout layer
model.add(Dropout(0.20))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


# Add batch normalization to your network

from keras.layers import BatchNormalization

# Add a convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Extracting a kernel from a trained network

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(5, kernel_size=2, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Add another convolutional layer
model.add(Conv2D(15, kernel_size=2, activation='relu', 
    input_shape=(img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Load the weights into the model
model.load_weights(path+'weights.hdf5')

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][:,:,0, 0]
print(kernel)

# Define convolution function
def convolution(image, kernel):
    kernel = kernel - kernel.mean()
    result = np.zeros(image.shape)

    for ii in range(image.shape[0]-2):
        for jj in range(image.shape[1]-2):
            result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)

    return result

# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)

# Visualize the result
plt.imshow(out)
plt.show()










