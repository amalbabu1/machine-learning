# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:18:30 2019

@author: student
"""

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
model = LogisticRegression(solver='liblinear')
rfe = RFE(model,4)
selector = rfe.fit(X,y)

print("num Feature: %d" % selector.n_features_)
print("selected Feature: %s" % selector.support_)
print("Feature ranking: %s" % selector.ranking_)
cols = selector.get_support(indices=True)
print(cols)
X_new = X[:,cols]


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_new,y)
y_pred = knn.predict(X_new)
print(metrics.accuracy_score(y,y_pred))
