#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(Xtrain, ytrain)
print("Logistc Regression Traning -> {:f}".format(lr.score(Xtrain, ytrain)))
print("Logistc Regression Traning -> {:f}".format(lr.score(Xtest, ytest)))

# Apply SVM and print scores
svm = SVC()
svm.fit(Xtrain, ytrain)
print("SVC Traning -> {:f}".format(svm.score(Xtrain, ytrain)))
print("SVC Traning -> {:f}".format(svm.score(Xtest, ytest)))
del Xtest, Xtrain, ytest, ytrain, digits

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

boston = datasets.load_boston()
X = boston.data
y = boston.target

# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_pred - y_i_true)**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)

del X, y, boston, w_fit

import numpy as np

# Cancer Breast dataset 
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# Mathematical functions for logistic and hinge losses
def log_loss(raw_model_output):
   return np.log(1+np.exp(-raw_model_output))

def hinge_loss(raw_model_output):
   return np.maximum(0,1-raw_model_output)

# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)

del X, y, w_fit, bc


import matplotlib.pyplot as plt

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

train_errs, test_errs = [], []

C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Loop over values of C
for C_value in C_values:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)
    
    # Evaluate error rates and append to lists
    train_errs.append(1.0 - lr.score(X_train, y_train))
    test_errs.append(1.0 - lr.score(X_test, y_test))
    
# Plot results
plt.semilogx(C_values, train_errs, C_values, test_errs)
plt.legend(("train", "validation"))
plt.show()

del C_value, C_values, train_errs, test_errs

import sys
sys.path.append("/home/esssfff/Documents/Github/Examples/Linear_Classifiers/")
from sklearn.model_selection import GridSearchCV
from utilities import plot_classifier

X_train, X_test, y_train, y_test = train_test_split(digits.data[:,0:2], digits.target)

# Specify L1 regularization
lr = LogisticRegression(penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))
plot_classifier(X_train, y_train, searcher, proba=True)

# Get predicted probabilities
proba = searcher.predict_proba(X_train)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba,axis=1))

# function to plot the imagem according de index
def show_digit(proba_inds):
    plt.gray()
    plt.matshow(digits.images[proba_inds])
    return plt.show()

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1])

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0])

del coefs, proba_inds, proba, digits


# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression()
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class="multinomial" ,solver="lbfgs")
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))

del X_test, X_train, y_test, y_train


wine = datasets.load_wine()
X, y = wine.data[:,0:2], wine.target

# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))

del X, X_small, y, y_small, wine

digits = datasets.load_digits()
X, y = digits.data, digits.target==2

# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)

# Report the best parameters
print("Best CV params", searcher.best_params_)

del X, y, parameters

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target==2)

# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

del parameters, digits

from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# We set random_state=0 for reproducibility 
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
             'loss':['hinge', 'log'], 'penalty':['l1', 'l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)
y_pred = searcher.predict(X_test)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

del X_train, X_test, y_train, y_test, y_pred, parameters