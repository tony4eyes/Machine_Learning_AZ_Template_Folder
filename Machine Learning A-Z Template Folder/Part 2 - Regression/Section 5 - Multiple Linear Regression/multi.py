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

import sklearn.model_selection as cv
X_train, X_test, Y_train, Y_test =  cv.train_test_split(X, Y, test_size = 0.2, random_state = 0)

import sklearn.linear_model as lm
regressor = lm.LinearRegression()
regressor.fit(X_train, Y_train)
regressor.predict(X_test)

import matplotlib.pyplot as plt
