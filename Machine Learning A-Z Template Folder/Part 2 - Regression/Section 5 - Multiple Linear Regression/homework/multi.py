# -*- coding: utf-8 -*-
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

import sklearn.preprocessing as pc

#LabelEncoder - convert categories to categorical integer
le = pc.LabelEncoder();
X[:,3] = le.fit_transform(X[:,3])

#OneHotEncoder - categorical integer
oe = pc.OneHotEncoder(categorical_features = [3]);
X = oe.fit_transform(X).toarray()


#split training and test set
import sklearn.model_selection as cv
X_train, X_test, Y_train, Y_test =  cv.train_test_split(X, Y, test_size = 0.2, random_state = 0)

import sklearn.linear_model as lm
regressor = lm.LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
import numpy as np

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:,range(np.size(X,1))]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary(),p