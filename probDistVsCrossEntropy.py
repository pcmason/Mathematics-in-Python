#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:53:52 2022

@author: paulmason
"""
'''
Binary classification problem with labels 0 and 1
There is a certain prob for one event and an impossible prob for the other event
Then Calculate cross entropy for different 'predicted' prob dists going from a perfect match
To the exact opposite prob dist
'''

#Import log for the cross entropy calculation
from math import log
#Import pyplot to show the change in cross entropy dependent upon 'predicted' prob dists
from matplotlib import pyplot

#Calculate cross entropy, ets ensures you do not calculate log(0)
def cross_entropy(p, q, ets = 1e-15):
    return -sum([p[i] * log(q[i] + ets) for i in range(len(p))])

#Define target dist for 2 events
target = [0.0, 1.0]
#Define probs for the first event
probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#Create prob dists for the 2 events
dists = [[1 - p, p] for p in probs]
#Calculate cross entropy for each dist
ents = [cross_entropy(target, d) for d in dists]

#Plot prob dist vs cross entropy
pyplot.plot([1 - p for p in probs], ents, marker = '.')
pyplot.title('Probability Distribution vs. Cross Entropy')
pyplot.xticks([1 - p for p in probs], ['[%.2f, %.2f]' % (d[0], d[1]) for d in dists], rotation = 70)
pyplot.subplots_adjust(bottom = 0.2)
pyplot.xlabel('Probability Distribution')
pyplot.ylabel('Cross Entropy (nats)')
pyplot.show()