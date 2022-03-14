#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:22:19 2022

@author: paulmason
"""
#Solving the birthday problem in Python
#Birthday problem - dependant on how many people are in a group, 
#how likely is it that 2 people in the group have the same birthday

#define group size
n = 50

#days in the year
days = 365

#calculate prob for different group sizes
p = 1.0
for i in range(1, n):
    av = days - i
    p *= av / days
    print('n=%d, %d/%d, p=%.3f, 1-p=%.3f' % (i + 1, av, days, p * 100, (1-p) * 100))