# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:43:22 2019

@author:_babu.jr_
"""
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y=array[:,8]
scalar=StandardScaler().fit(X)
rescaledX = scalar.transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])
