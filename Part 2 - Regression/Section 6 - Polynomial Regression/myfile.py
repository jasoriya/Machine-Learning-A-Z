# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# Importing the dataset
# =============================================================================
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
# =============================================================================
# Fitting the linear regressor to the dataset
# =============================================================================
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
# =============================================================================
# Fiting the polynomial regressor to the dataset 
# =============================================================================
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)
# =============================================================================
# Visualizing the Linear regression results 
# =============================================================================
plt.scatter(X, Y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
# =============================================================================
# Visualizing the Polynomial regression results
# =============================================================================
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
# =============================================================================
# Predicting a new result with Linear Regression
# =============================================================================
linear_regressor.predict(6.5)
# =============================================================================
# Predicting a new result with Linear Regression
# =============================================================================
lin_reg_2.predict(poly_reg.fit_transform(6.5))
