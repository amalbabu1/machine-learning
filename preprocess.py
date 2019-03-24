# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:59:25 2019

@author: _babu.jr_
"""
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from numpy import set_printoptions
filename = 'pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
array = dataframe.values
X = array[:,0:8]
y=array[:,8]
scalar = MinMaxScaler(feature_range=(0,1))
rescaledX=scalar.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])
