# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:31:59 2019

@author: student
"""

from pandas import read_csv
from sklearn.decomposition import PCA
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
pca = PCA(n_components=3)
fit = pca.fit(X)
print(fit.explained_variance_ratio_)
print(fit.components_)