# -*- coding: utf-8 -*-
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

import sklearn.preprocessing as pc

le = pc.LabelEncoder();
X[:,3] = le.fit_transform(X[:,3])

#OneHotEncoder - categorical integer
oe = pc.OneHotEncoder(categorical_features = [3]);
X = oe.fit_transform(X).toarray()