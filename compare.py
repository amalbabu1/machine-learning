# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:08:09 2019

@author: _babu.jr_
"""
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
models = []
models.append(('LR',LogisticRegression(solver='liblinear')))
models.append(('KNN1',KNeighborsClassifier(n_neighbors=1)))
models.append(('KNN5',KNeighborsClassifier(n_neighbors=5)))
models.append(('Tree',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
        kfold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(model, X, y,cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)"%(name, cv_results.mean(), cv_results.std())
        print(msg)
        
        