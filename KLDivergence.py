#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:18:24 2022

@author: paulmason
"""
#Worked example of KL Divergence

#Import pyplot for plots
from matplotlib import pyplot
#Import log2 for the KL Divergence method
from math import log

#Calculate KL Divergence
def kl_divergence(p, q):
    #Use natural log to return answer in nats
    return sum(p[i] * log(p[i] / q[i]) for i in range(len(p)))

#Random variable with 3 different events as colors
events = ['red', 'green', 'blue']
#2 different prob dists for this var
p = [0.1, 0.4, 0.5]
q = [0.8, 0.15, 0.05]
#Show probs add up to 1
print('P = %.3f, Q = %.3f' % (sum(p), sum(q)))

#Plot first dist
pyplot.subplot(2, 1, 1)
pyplot.bar(events, p)

#Plot second dist
pyplot.subplot(2, 1, 2)
pyplot.bar(events, q)

#Show the plot
pyplot.show()

#Calculate (P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f nats' % kl_pq)

#Calculate (Q || P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f nats' % kl_qp)

#Import scipy's method to calc KL Divergence
from scipy.special import rel_entr

#Calculate (P || Q)
sci_kl_pq = rel_entr(p, q)
print('KL(P || Q): %.3f nats' % sum(sci_kl_pq))

#Calculate (Q || P)
sci_kl_qp = rel_entr(q, p)
print('KL(Q || P): %.3f nats' % sum(sci_kl_qp))



