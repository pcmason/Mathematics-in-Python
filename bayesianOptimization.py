#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:42:03 2022

@author: paulmason
"""
#Define a test problem to run Bayesian Optimization on
#Will use a multimodal problem with 5 peaks calculated below

#Import methods for objective function
from numpy.random import normal, random
from math import sin, pi
#Objective function
def objective(x, noise = 0.1):
    #Add a positive/negative value to the evaluation to make it more challenging to optimize
    noise = normal(loc = 0, scale = noise)
    return (x**2 * sin(5 * pi * x)**6.0) + noise

#Import numpy
import numpy as np
#Test the function with grid-based sample of the domain [0, 1]
x = np.arange(0, 1, 0.01)

#Sample the domain without noise
y = [objective(i, 0) for i in x]

#Sample the domain with noise
yNoise = np.asarray([objective(i) for i in x])

#Find the best result
ix = np.argmax(y)
print('Optima: x = %.3f, y = %.3f' % (x[ix], y[ix]))

#Import library for plotting
from matplotlib import pyplot

#Plot the points with noise (blue dots on graph)
pyplot.scatter(x, yNoise)

#Plot the points without noise (orange line on graph)
pyplot.scatter(x, y)

#Show the plot
pyplot.show()

#Reshape data into rows and columns
x = x.reshape(len(x), 1)
yNoise = yNoise.reshape(len(yNoise), 1)

#Import GP regressor method from sklearn
from sklearn.gaussian_process import GaussianProcessRegressor

#Define the model
model = GaussianProcessRegressor()

#Fit the model
model.fit(x, yNoise)

#Get the mean and std from the model at a given point
yhat = model.predict(x, return_std = True)

#Import warnings module for surrogate method
import warnings

#Surrogate for the objective function
def surrogate(model, x):
    #Catch any warnings generated when making a prediction
    with warnings.catch_warnings():
        #Ignore generated warnings
        warnings.simplefilter("ignore")
        return model.predict(x, return_std = True)
    
#Plot real Observations vs surrogate function
def plot(x, y, model):
    #Scatter plot of inputs and real objective function
    pyplot.scatter(x, y)
    #Line plot of surrogate function across domain
    xSamples = np.asarray(np.arange(0, 1, 0.001))
    xSamples = xSamples.reshape(len(xSamples), 1)
    ySamples, _ = surrogate(model, xSamples)
    pyplot.plot(xSamples, ySamples)
    #Show the plot
    pyplot.show()
    
#Now finally plot the real observations against the surrogate function
plot(x, yNoise, model)

#Optimize the acquisition function
def opt_acquisition(x, y, model):
    #Random search generate Random samples
    xSamples = random(100)
    xSamples = xSamples.reshape(len(xSamples), 1)
    #Calculate the acquisition function for each samples
    scores = acquisition(x, xSamples, model)
    #Locate the index of the largest scores
    ix = np.argmax(scores)
    return xSamples[ix, 0]

#Import library for acquisition method
from scipy.stats import norm

#Probability of improvement (PI) acquisition function
def acquisition(x, xSamples, model):
    #Calculate the best surrogate score found so far
    yhat, _ = surrogate(model, x)
    best = max(yhat)
    #Calculate mean and std through surrogate function
    mu, std = surrogate(model, xSamples)
    mu = mu[:, 0]
    #Calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs

#Perform the optimization process
for i in range(100):
    #Select the next point to sample
    nextPoint = opt_acquisition(x, yNoise, model)
    #Sample the point
    actual = objective(nextPoint)
    #Summarize the finding for reporting
    est, _ = surrogate(model, [[nextPoint]])
    print('>x = %.3f, f() = %.3f, actual = %.3f' % (nextPoint, est, actual))
    #Add the data to the dataset
    x = np.vstack((x, [[nextPoint]]))
    yNoise = np.vstack((yNoise, [[actual]]))
    #Update the model
    model.fit(x, yNoise)
    
#Plot all the samples and the final surrogate function
#You can see the sampling works with many points graphed around the optima
plot(x, yNoise, model)
#Best result
ix = np.argmax(yNoise)
print('Best result: x = %.3f, y = %.3f' % (x[ix], yNoise[ix]))

    
    

