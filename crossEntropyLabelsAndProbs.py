#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:20:19 2022

@author: paulmason
"""
#Use cross entropy as a loss method with a binary classification task

#Import log for the cross entropy method
from math import log
#Import mean to get average of the cross entropy on the example
from numpy import mean

#Define 10 actual class labels 
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#Define 10 predicted class labels
q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]

#Function to calculate cross entropy
def cross_entropy(p, q):
    return -sum([p[i] * log(q[i]) for i in range(len(p))])

#Calculate cross entropy for each example
results = list()
for i in range(len(p)):
    #Create dist for each event {0, 1} 
    expected = [1 - p[i], p[i]]
    predicted = [1 - q[i], q[i]]
    #Calculate, output and append the cross entropy for the 2 events
    ce = cross_entropy(expected, predicted)
    #Print out results for each iteration for y and yhat
    print('>[y = %.2f, yhat = %.2f] ce: %.3f nats' % (p[i], q[i], ce))
    results.append(ce)
    
#Calculate and output the average cross entropy
mean_ce = mean(results)
print('Average cross entropy: %.3f nats' % mean_ce)

#Use scikit-learn log-loss method which should be equal to the cross entropy
from sklearn.metrics import log_loss

#Define prob for each event {0, 1}
y_true = [[1 - v, v] for v in p]
y_pred = [[1 - v, v] for v in q]

#Calculate and output the Average log loss (should be same as cross entropy)
ll = log_loss(y_true, y_pred)
print('Average Log Loss: %.3f' % ll)

#Confirm above calculations using the methods from Keras and TensorFlow
from numpy import asarray
from keras import backend
from keras.losses import binary_crossentropy

#Use asarray with the same data from above
p = asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
q = asarray([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])

#Convert them to keras variables using the backend method
y_true = backend.variable(p)
y_pred = backend.variable(q)

#Calculate and outpute the average cross entropy using keras (result should be same as above)
keras_mean_ce = backend.eval(binary_crossentropy(y_true, y_pred))
print('Average cross entropy: %.3f nats'  % keras_mean_ce)

