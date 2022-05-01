#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:52:47 2022

@author: paulmason
"""
#Import sklearn to generate problem and for optimization
import sklearn
from sklearn.datasets import make_blobs
#Import skopt for optimization
import skopt
#Import numpy
import numpy as np

#Generate 2d classification dataset
x, y = make_blobs(n_samples = 500, centers = 3, n_features = 2)
#Define the model
model = sklearn.neighbors.KNeighborsClassifier()

#Define the space of hyperparameters to search
search_space = [skopt.space.Integer(1, 5, name = "n_neighbors"), skopt.space.Integer(1, 2, name = "p")]


from skopt.utils import use_named_args
#Define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
    #Set parameters to something
    model.set_params(**params)
    #Calculate the 5-fold cross-validation
    result = sklearn.model_selection.cross_val_score(model, x, y, cv = 5, n_jobs = 1, scoring = "accuracy")
    #Calculate the mean of the scores
    estimate = np.mean(result)
    return 1.0 - estimate

#Perform optimization
result = skopt.gp_minimize(evaluate_model, search_space)

#Summarize findings
print('Best accuracy: %.3f' % (1 - result.fun))
print('Best parameters: n_neighbors = %d, p = %d' % (result.x[0], result.x[1]))

