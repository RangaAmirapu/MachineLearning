# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:55:24 2019

@author: Ranga Rao Amirapu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset  = pd.read_csv('Salary_Data.csv')

#metrics -X :indpendent variables and features -Y: Dependent variable
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#splitting into training set and test set
from sklearn.model_selection import train_test_split
#random state is used to expect same results while learning, In a real case remove this
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size = 1/3, random_state = 0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

#predicting the test set results
yPredicted  = regressor.predict(xTest)

#Visualizing training set results
plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, regressor.predict(xTrain), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing test set results
plt.scatter(xTest, yTest, color = 'red')
plt.plot(xTrain, regressor.predict(xTrain), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
