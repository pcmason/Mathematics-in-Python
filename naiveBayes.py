#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 18:15:40 2022

@author: paulmason
"""
###Worked examples using Naive Bayes 

#First generate binary classification prob from scikit-learn
from sklearn.datasets import make_blobs
import numpy as np
from statistics import NormalDist

#Fit a probability distribution to a univariate data sample
def fit_distribution(data):
    #Estimate parameters
    mu = np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    
    #Fit distribution
    dist = NormalDist(mu, sigma)
    return dist

#Calculate the independent conditional probability
def probability(x, prior, dist1, dist2):
    return prior * dist1.pdf(x[0]) * dist2.pdf(x[1])

#Generate 2D classification dataset with 100 observations
#random_state is set to 1 to ensure a random sample is generated every time code is run
x, y = make_blobs(n_samples = 100, centers = 2, n_features = 2, random_state = 1)

#print out summary statistics
print(x.shape, y.shape, '\n')
#print out first five values for x and y
print(x[:5], '\n')
print(y[:5], '\n')

#Sort the data into classes
xy0 = x[y == 0]
xy1 = x[y == 1]
print(xy0.shape, xy1.shape, '\n')

#Calculate the priors (will be 50% since there is the same # of examples in each class)
prior0 = len(xy0) / len(x)
prior1 = len(xy1) / len(x)
print(prior0, prior1, '\n')

#Create PDFs for y = 0
x1y0 = fit_distribution(xy0[:, 0])
x2y0 = fit_distribution(xy0[:, 1])

#Create PDFs for y = 1
x1y1 = fit_distribution(xy1[:, 0])
x2y1 = fit_distribution(xy1[:, 1])
print('\n')

#Classify the first example from the dataset
xSample, ySample = x[0], y[0]

#Calculate score of example belonging to the first class
py0 = probability(xSample, prior0, x1y0, x2y0)
#Calculate score of example belonging to the second class
py1 = probability(xSample, prior1, x1y1, x2y1)
#Print out the scores
print('P(y=0 | %s) = %.3f' % (xSample, py0 * 100), '\n')
print('P(y=1 | %s) = %.3f' % (xSample, py1 * 100), '\n')
#print out what y actually is
print('Truth: y = %d' % ySample, '\n')






    