import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset  = pd.read_csv('Data.csv')

#metrics -X :indpendent variables and features -Y: Dependent variable
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Fix missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Categorical data encoding and Dummy encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#The 'categorical_features' keyword is deprecated in version 0.20 and will be removed in 0.22. You can use the ColumnTransformer instead.
#labelEncoder_x = LabelEncoder()
#x[:,0] = labelEncoder_x.fit_transform(x[:,0])
#oneHotEncoder = OneHotEncoder(categorical_features=[0])
#x  = oneHotEncoder.fit_transform(x).toarray()

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x  =  np.array(ct.fit_transform(x), dtype=np.float)

yLabelEncoder = LabelEncoder()
y = yLabelEncoder.fit_transform(y)

#Splitting into training set and test set
from sklearn.model_selection import train_test_split
#random state is used to expect same results while learning, In a real case remove this
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Feature Scaling
#2types - standardisation and normalisation
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xTrain = sc_x.fit_transform(xTrain)
xTest = sc_x.transform(xTest)






