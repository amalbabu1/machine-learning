# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:59:34 2019

@author: _babu.jr_
"""
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model,X,y,cv=kfold)
print("accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0 , results.std()*100))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
results = cross_val_score(knn,X,y,cv=kfold)
print("accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0 , results.std()*100))
