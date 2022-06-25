#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:30:24 2022

@author: paulmason
"""
#Worked example calculating cross-entropy

#Import pyplot for plots
from matplotlib import pyplot
#Import log2 for the cross entropy method
from math import log2

#Calculate cross entropy in bits
def cross_entropy(p, q):
    return -sum([p[i] * log2(q[i]) for i in range(len(p))])

#Calculate KL Divergence in bits
def kl_divergence(p, q):
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))

#Calculate entropy for a given prob dist
def entropy(p):
    return -sum([p[i] * log2(p[i]) for i in range(len(p))])

#Calculate cross entropy using kl and entropy
def cross_entropy_kl(p, q):
    return entropy(p) + kl_divergence(p, q)

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

#Calculate cross entorpy H(p, q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)

#Calculate cross entropy H(q, p)
ce_qp = cross_entropy(q, p)
print('H(Q, P): %.3f bits' % ce_qp)

#Calculate cross entropy H(p, p) == entropy(p)
ce_pp = cross_entropy(p, p)
print('H(P, P): %.3f bits' % ce_pp)

#Calculate cross entropy H(q, q) == entropy(q)
ce_qq = cross_entropy(q, q)
print('H(Q, Q): %.3f bits' % ce_qq)

#Calculate H(p) entropy
en_p = entropy(p)
print('H(P): %.3f bits' % en_p)

#Calculate kl divergence
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)

#Calculate cross entropy using KL div
ce_pq_kl = cross_entropy_kl(p, q)
print('H(P, Q): %.3f bits' % ce_pq_kl)

