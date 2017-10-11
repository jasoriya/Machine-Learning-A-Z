# -*- coding: utf-8 -*-

#Importing the libraries
import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

#Taking care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features= [-1])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

#Splitting the data into train & dev sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Implementing the Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Making Predictions 
y_pred = regressor.predict(X_test)

#OLSR / Backward elimination
import statsmodels.api as sm
X = np.append(np.ones((50, 1)).astype(int), X, axis = 1)
X_opt = X
for column in range(X.shape[1]):
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    if ((regressor_OLS.pvalues > 0.05).any()):
        X_opt = np.delete(X_opt, np.argmax(regressor_OLS.pvalues), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
y_opt = regressor_OLS.predict(X_test)


