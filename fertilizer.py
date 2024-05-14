import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
df_cotton=pd.read_csv('Unprocessed Data.csv')
print(df_cotton)


df_cotton=df_cotton.fillna(df_cotton.groupby('State').transform('mean'))

print(df_cotton)

mapping = ({'Alabama':1,
'Arizona':2,
'Arkansas':3,
'California':4,
'Georgia':5,
'Louisiana':6,
'Mississippi':7,
'Missouri':8,
'New Mexico':9,
'North Carolina':10,
'Oklahoma':11,
'South Carolina':12,
'Tennessee':13,
'Texas':14,
           })
df_cotton=df_cotton.replace({'State': mapping})
x=df_cotton.drop({'Nitrogen (Pounds/Acre)','Phosphorous (Pounds/Acre)','Potash (Pounds/Acre)'},axis=1)
y=df_cotton.drop({'State','Year','Area Planted (acres)','Harvested Area (acres)','Lint Yield (Pounds/Harvested Acre)','Nitrogen (%)','Phosphorous (%)','Potash (%)'},axis=1)
print(x)

print(y)

from sklearn.datasets import make_regression
print(x.shape, y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,z_train,z_test=train_test_split(x,y,test_size=0.2) #80% for Training and 20% for Testing
print(x_train.shape,x_test.shape,z_train.shape,z_test.shape)
from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
estimator = MultiOutputRegressor(ensemble.GradientBoostingRegressor())
estimator.fit(x_train,z_train)
print(estimator.score(x_test,z_test))

print(estimator.score(x_train,z_train))


n_input=[[1,1964,99.00000,100.000000,99.0,419.355556,407.822222,644.688889]]
n_output=estimator.predict(n_input)
print(n_output)

