#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:13:03 2022

@author: paulmason
"""
#worked example using GaussianNB from sklearn

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

#Generate 2D Classification dataset
x, y = make_blobs(n_samples = 100, n_features = 2, centers = 2, random_state = 1)

#Define the model
model = GaussianNB()

#Fit the model
model.fit(x, y)

#Select a single sample
xSample, ySample = [x[0]], y[0]

#Make a probabilistic prediction
yhat_prob = model.predict_proba(xSample)
print('Predicted probabilities: ', yhat_prob)

#Make a classification prediction
yhat_class = model.predict(xSample)
print('Predicted class: ', yhat_class)
print('Truth: y = %d' % ySample)

