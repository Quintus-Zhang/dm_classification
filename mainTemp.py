#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:16:35 2017

@author: Quintus
"""

import datetime as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from funcs import dataHandler, euclideanDistance, majorityVote, distanceWeighted

# params setup
start = dt.datetime(1998, 11, 22) 
end = dt.datetime(2017, 3, 20)
symbolList = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']
ind = '%5EGSPC'

indRtn = dataHandler(ind, start, end)
dates  = dataHandler(ind, start, end, True)
dates_ = list()
for i in reversed(range(len(dates))):
    dates_.append(dt.datetime.strptime(dates[i], '%Y-%m-%d'))

# sector indices' return
rtns = list()
for symbol in symbolList:
    rtns.append(dataHandler(symbol, start, end))
rtns = np.array(rtns).T

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
        
# calculate cumulative return
cumRtn = np.cumprod(indRtn[101:] * np.array(signal) + 1)

#==============================================================================
# plotting-1
#==============================================================================
time = matplotlib.dates.date2num(dates_[102:])
plt.figure()
plt.plot_date(time, cumRtn, '-')
plt.plot_date(time, np.cumprod(indRtn[101:] + 1), '-')
plt.legend(['Strategy', 'S&P500'], loc = 4)
plt.xlabel('Time')
plt.ylabel('Cumulative return')
plt.show()

time = matplotlib.dates.date2num(dates_[backtestStart+2:])
plt.figure()
plt.plot_date(time, cumRtn, '-', c='dodgerblue')
plt.plot_date(time, np.cumprod(indRtn[backtestStart+1:] + 1), '-', c='orange')
plt.grid(True, which='major', c='lightslategrey', linestyle='-', linewidth=0.5)
plt.legend(['Strategy', 'S&P500'], loc = 4, frameon=True, fontsize='medium')
plt.xlabel('Time')
plt.ylabel('Cumulative return')
plt.show()

#==============================================================================
# plotting-2
#==============================================================================
plt.figure()
time = matplotlib.dates.date2num(dates_[:-1])
sectorCumRtns = np.cumprod(rtns + 1, axis=0)
for i in range(rtns.shape[1]):
    plt.plot_date(time, sectorCumRtns[:, i], '-')
plt.legend(['Consumer Discretionary', 'Consumer Staples','Energy', 'Financials', 'Health Care', 'Industrials', 'Materials', 'Technology', 'Utilities'], loc = 2, fontsize='medium')
plt.xlabel('Time')
plt.ylabel('Cumulative return')
plt.show()

    








