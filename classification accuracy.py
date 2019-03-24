# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:43:36 2019

@author: _babu.jr_
"""
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=4)
model = LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test,y_pred)
print(matrix)