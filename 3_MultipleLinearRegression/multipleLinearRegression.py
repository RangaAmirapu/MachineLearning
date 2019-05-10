# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:55:24 2019

@author: Ranga Rao Amirapu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset  = pd.read_csv('50_Startups.csv')

#metrics -X :indpendent variables and features -Y: Dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Categorical data encoding and Dummy encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    
    remainder='passthrough'                         
)
X  =  np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the dummy variable trap This will managed by library 
X= X[: , 1:]



#splitting into training set and test set
from sklearn.model_selection import train_test_split
#random state is used to expect same results while learning, In a real case remove this
XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(XTrain, yTrain)

#Predicting the test set results
yPred  = regressor.predict(XTest)

#Building the optimal model with backward elimination
import statsmodels.formula.api as sm
#Stats model doesnt include teh X0 which is required for calculating the coeff where as linear regression lib does have
X = np.append(arr = np.ones(shape = (50,1)).astype(int),axis = 1, values = X)
#XOptimal will have only indepenedent variables of high statistical significance

#TODO: convert to for loop

XOptimal = X[:, [0, 1, 2, 3, 4, 5]]
regressorOls = sm.OLS(endog = y, exog = XOptimal).fit()
regressorOls.summary()

XOptimal = X[:, [0, 1, 3, 4, 5]]
regressorOls = sm.OLS(endog = y, exog = XOptimal).fit()
regressorOls.summary()

XOptimal = X[:, [0, 3, 4, 5]]
regressorOls = sm.OLS(endog = y, exog = XOptimal).fit()
regressorOls.summary()

XOptimal = X[:, [0, 3, 5]]
regressorOls = sm.OLS(endog = y, exog = XOptimal).fit()
regressorOls.summary()

XOptimal = X[:, [0, 3]]
regressorOls = sm.OLS(endog = y, exog = XOptimal).fit()
regressorOls.summary()
































