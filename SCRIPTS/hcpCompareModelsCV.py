# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 19:40:39 2019

@author: SalmanKarim
"""

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.svm import SVR




#import matplotlib.pyplot as pyplot



#read the file
df = read_csv('FenModified.csv')
#setting index

df.index = df.Date
df.drop('Date', axis=1, inplace=True)


array = df.values
X = array[:,0:6]
Y = array[:,6]

# prepare models
models = []
models.append(('LR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
#models.append(('SVM', SVR()))

print("Mean Squared Error:") 
# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
      kfold = KFold(n_splits=10, random_state=7)
      cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
      results.append(cv_results)
      names.append(name)
      msg = ("%s: [ MSE: %0.3f ]  [ std (%0.3f)]" % (name, cv_results.mean(), cv_results.std()))

      
      print(msg)
      
print("------------------------------------------------")      
print("------------------------------------------------")  

      
print("Mean Absolute Error:")      
scoring1 = 'neg_mean_absolute_error'
for name, model in models:
      kfold = KFold(n_splits=10, random_state=7)
      cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring1)
      results.append(cv_results)
      names.append(name)
      msg = ("%s: [ MAE:  %0.3f ]  [ std (%0.3f)]" % (name, cv_results.mean(), cv_results.std()))

      
      print(msg)
print("------------------------------------------------")      
print("------------------------------------------------")  
            
print("R Squared Error:")      
scoring2 = 'r2'
for name, model in models:
      kfold = KFold(n_splits=10, random_state=7)
      cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring2)
      results.append(cv_results)
      names.append(name)
      msg = ("%s: [ R2:  %0.3f ]  [ std (%0.3f)]" % (name, cv_results.mean(), cv_results.std()))

      
      print(msg)     
      
print("------------------------------------------------")      
print("------------------------------------------------")  
            
print("explained_variance:")      
scoring3 = 'explained_variance'
for name, model in models:
      kfold = KFold(n_splits=10, random_state=7)
      cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring3)
      results.append(cv_results)
      names.append(name)
      msg = ("%s: [ V:  %0.3f ]  [ std (%0.3f)]" % (name, cv_results.mean(), cv_results.std()))

      
      print(msg)    


print(X.shape)

print(Y.shape)
print(df.shape)      
    
# boxplot algorithm comparison
#fig = pyplot.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()


