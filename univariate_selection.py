# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 09:58:02 2019

@author: _babu.jr_
"""
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,y)
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:])

