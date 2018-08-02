# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:,:-1]
Y = data.iloc[:,1]

import sklearn.cross_validation as cv
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size = 1/3, random_state = 0)

import sklearn.linear_model as lr
regressor = lr.LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()