# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import datetime as dt
import urllib
import pandas as pd
import numpy as np

def dataHandler(symbol, start, end, reportDate=False):
    url = 'https://chart.finance.yahoo.com/table.csv?'+ \
          's=' + symbol + \
          '&a=' + str(start.month) + \
          '&b=' + str(start.day) + \
          '&c=' + str(start.year) + \
          '&d=' + str(end.month) + \
          '&e=' + str(end.day) + \
          '&f=' + str(end.year) + '&g=m&ignore=.csv'
    
    # retrieve data from yahoo finance and store in dataframe
    html = urllib.request.urlopen(url).read()
    string = html.decode("utf-8")
    strList = string.split('\n')
    listofLists = list()
    for strs in strList:
        if strs:
            listofLists.append(strs.split(','))
    
    data = pd.DataFrame(listofLists[1:], columns=listofLists[0])
    
    if reportDate:
        return data['Date'].copy().values
    
    data['Date'] = data['Date'].apply(dt.datetime.strptime, args=('%Y-%m-%d',))
    data.sort_values('Date', inplace=True)
    data.set_index('Date', drop=True, inplace=True)
    data = data.astype(float, copy=True)
    rtn = data.ix[1:, 'Adj Close'].values / data.ix[0:-1, 'Adj Close'].values - 1
    return rtn

# Euclidean distance
def euclideanDistance(a, b):
    return np.sqrt(np.sum((a - b)**2))
 
# Majority Voting
def majorityVote(rtns, x, indRtn, n):
    vote = np.zeros([n, 2])
    for i in range(x):
        if i < n:
            vote[i, 0] = euclideanDistance(rtns[x, :], rtns[i, :])
            vote[i, 1] = indRtn[i+1]
        else:
            vote = vote[vote[:, 0].argsort()]
            if vote[-1, 0] > euclideanDistance(rtns[x, :], rtns[i, :]):
                vote[-1, 0] = euclideanDistance(rtns[x, :], rtns[i, :])
                vote[-1, 1] = indRtn[i+1]
    prob = np.sum(vote[:, -1]>0) / n
    return prob

# Distance Weighted Voting
def distanceWeighted(rtns, x, indRtn, n):
    vote = np.zeros([n, 2])
    for i in range(x):
        if i < n:
            vote[i, 0] = euclideanDistance(rtns[x, :], rtns[i, :])
            vote[i, 1] = indRtn[i+1]
        else:
            vote = vote[vote[:, 0].argsort()]
            if vote[-1, 0] > euclideanDistance(rtns[x, :], rtns[i, :]):
                vote[-1, 0] = euclideanDistance(rtns[x, :], rtns[i, :])
                vote[-1, 1] = indRtn[i+1]
    longScore = np.sum(1 / vote[vote[:, -1] > 0, 0]**2)
    shortScore = np.sum(1 / vote[vote[:, -1] < 0, 0]**2)
    return longScore - shortScore

    
    
    
    