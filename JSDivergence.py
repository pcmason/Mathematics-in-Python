#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:54:09 2022

@author: paulmason
"""
#Worked example of JS Divergence

#Import log 2 from math for JS and KL divergence methods and sqrt to calc distance
from math import log2, sqrt
#Import asarray from numpy for the different prob dists
from numpy import asarray

#Calculate KL divergence in bits
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))

#Calculate the JS divergence in bits
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

#Random variable with 3 different events as colors
events = ['red', 'green', 'blue']
#2 different prob dists for this var
p = asarray([0.1, 0.4, 0.5])
q = asarray([0.8, 0.15, 0.05])

#Calculate JS(P || Q)
js_pq = js_divergence(p, q)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
print('JS(P || Q) distance: %.3f' % sqrt(js_pq))

#Calculate JS(Q || P)
js_qp = js_divergence(q, p)
print('JS(Q || P) divergence: %.3f bits' % js_qp)
print('JS(Q || P distance: %.3f' % sqrt(js_qp))

#Import the scipy library method for JS
from scipy.spatial.distance import jensenshannon

#Calculate JS(P || Q)
sci_js_pq = jensenshannon(p, q, base = 2)
print('JS(P || Q) distance: %.3f' % sci_js_pq)

#Calculate JS(Q || P)
sci_js_qp = jensenshannon(q, p, base = 2)
print('JS(Q || P) distance: %.3f' % sci_js_qp)



