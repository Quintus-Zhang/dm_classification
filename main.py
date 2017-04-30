#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:32:45 2017

@author: Quintus
"""
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from funcs import dataHandler, euclideanDistance, majorityVote, distanceWeighted

# params setup
start = dt.datetime(1998, 11, 22) 
end = dt.datetime(2017, 3, 20)
symbolList = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
ind = '%5EGSPC'

# sector indices' return
rtns = list()
for symbol in symbolList:
    rtns.append(dataHandler(symbol, start, end))
rtns = np.array(rtns).T
indRtn = dataHandler(ind, start, end)

# ranking
ranking = np.array([(-a).argsort().argsort()+1 for a in rtns])

# logistic regression
X_train = ranking[0:100, :]
y_train = np.sign(indRtn[1:101])

sample_weight = 100 * np.abs(np.random.randn(100))
sample_weight[y_train == 1] += 10

lr = LogisticRegression(C=100.0, random_state=0, solver='newton-cg')
lr.fit(X_train, y_train, sample_weight = sample_weight)
signal = lr.predict(rtns[100:-1, :])

# 
#svm = SVC(kernel='rbf', C=1000.0, random_state=0, )
#svm.fit(X_train, y_train)
#signal = svm.predict(rtns[100:-1, :])


# signals generation
n = 5
signal = list()
for x in range(100, 219):
    if distanceWeighted(rtns, x, indRtn, n) > 0.1:
        signal.append(1)
    elif distanceWeighted(rtns, x, indRtn, n) < 0.1: 
        signal.append(-1)
    else:
        signal.append(0)

#p = list()
#for i in range(100, 219):
#    if (np.sign(indRtn[i+1]) == signal[i-100]):
#        p.append(1)
#    else:
#        p.append(0)

# calculate cumulative return
cumRtn = np.cumprod(indRtn[101:] * np.array(signal) + 1)

#==============================================================================
# plotting
#==============================================================================
plt.figure()
plt.plot(cumRtn)
plt.plot(np.cumprod(indRtn[101:] + 1))
plt.legend(['Strategy', 'S&P500'], loc = 4)
plt.xlabel('Time')
plt.ylabel('Cumulative return')
plt.show()