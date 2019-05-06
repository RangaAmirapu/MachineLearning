import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset  = pd.read_csv('Data.csv')

#metrics -X :indpendent variables and features -Y: Dependent variable
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#splitting into training set and test set
from sklearn.model_selection import train_test_split
#random state is used to expect same results while learning, In a real case remove this
xTrain, xTest, yTrain, yTest = train_test_split = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Feature Scaling - do if required
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#xTrain = sc_x.fit_transform(xTrain)
#xTest = sc_x.transform(xTest)