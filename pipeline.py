# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:09:12 2019

@author: _babu.jr_
"""
from pandas import read_csv
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]

estimator =[]
estimator.append(('standardize',StandardScaler()))
estimator.append(('knn',KNeighborsClassifier(n_neighbors=5)))
model=Pipeline(estimator)

kfold =KFold(n_splits=10, random_state=7)
result = cross_val_score(model,X,y,cv=kfold)
print(result.mean())
print(result.std())