# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:33:40 2019

@author: Izadi"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.getcwd()
os.chdir(r'D:\desktop\Python_DM_ML_BA\ML\polynomial') 
dg = pd.read_csv('svm.txt', sep=',')
dg.head()
dg.columns
X = dg.drop(['position','level'], axis=1)
y = dg['level'] 

from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression()
linear_reg1.fit(X,y)
y_pred = linear_reg1.predict(X)
from sklearn.preprocessing import PolynomialFeatures
poly2_reg =  PolynomialFeatures(degree=2)
X_poly2 = poly2_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly2, y)
y_pred2 = linear_reg2.predict(X_poly2)
poly3_reg=  PolynomialFeatures(degree=3)
X_poly3= poly3_reg.fit_transform(X)
linear_reg3 = LinearRegression()
linear_reg3.fit(X_poly3,y)
y_pred3 = linear_reg3.predict(X_poly3)
poly4_reg=  PolynomialFeatures(degree=4)
X_poly4= poly4_reg.fit_transform(X)
linear_reg4 = LinearRegression()
linear_reg4.fit(X_poly4,y)
y_pred4 = linear_reg4.predict(X_poly4)
#plt.xlim([1, 10])
#plt.ylim([100, 200])
plt.scatter(X,y, color='red', s=100)
plt.plot(X,y_pred, color='blue')
plt.show()
plt.scatter(X,y, color='green',s=100)
plt.plot(X,y_pred2, color='purple')
plt.show()
plt.scatter(X,y, c='red', s=100)
plt.plot(X,y_pred3, c='olive')
plt.show()

df = pd.read_csv('salary.txt', sep =',')
df.head()
X = df.drop(['Position','Salary'], axis=1)
y = df.Salary

from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression()
linear_reg1.fit(X,y)
y_pred = linear_reg1.predict(X)
from sklearn.preprocessing import PolynomialFeatures
poly2_reg =  PolynomialFeatures(degree=2)
X_poly2 = poly2_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly2, y)
y_pred2 = linear_reg2.predict(X_poly2)
poly3_reg=  PolynomialFeatures(degree=3)
X_poly3= poly3_reg.fit_transform(X)
linear_reg3 = LinearRegression()
linear_reg3.fit(X_poly3,y)
y_pred3 = linear_reg3.predict(X_poly3)
poly4_reg=  PolynomialFeatures(degree=4)
X_poly4= poly4_reg.fit_transform(X)
linear_reg4 = LinearRegression()
linear_reg4.fit(X_poly4,y)
y_pred4 = linear_reg4.predict(X_poly4)

plt.scatter(X,y, s=100)
plt.plot(X,y_pred, color='blue')
plt.plot(X,y_pred2, color='purple')
plt.plot(X,y_pred3, c='red')
plt.plot(X,y_pred4, c='green')
plt.xlabel('Level', size=25)
plt.ylabel('Salary', size=25)
plt.title('Polynomial Plot', size=40)
plt.legend(['Deg=1', 'Deg=2', 'Deg=3', 'Deg=4'], loc=2)
plt.show()

dg = pd.read_csv('oceans.csv', skiprows=6)
dg.head()
dg.columns
X = dg.drop(['year'], axis=1)
y = dg.year/1000

from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression()
linear_reg1.fit(X,y)
y_pred = linear_reg1.predict(X)
from sklearn.preprocessing import PolynomialFeatures
poly2_reg =  PolynomialFeatures(degree=2)
X_poly2 = poly2_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly2, y)
y_pred2 = linear_reg2.predict(X_poly2)
poly3_reg=  PolynomialFeatures(degree=3)
X_poly3= poly3_reg.fit_transform(X)
linear_reg3 = LinearRegression()
linear_reg3.fit(X_poly3,y)
y_pred3 = linear_reg3.predict(X_poly3)
poly4_reg=  PolynomialFeatures(degree=4)
X_poly4= poly4_reg.fit_transform(X)
linear_reg4 = LinearRegression()
linear_reg4.fit(X_poly4,y)
y_pred4 = linear_reg4.predict(X_poly4)
#plt.xlim([1, 10])
#plt.ylim([100, 200])
plt.scatter(X,y, color='red', s=100)
plt.plot(X,y_pred, color='blue')
plt.show()
plt.scatter(X,y, color='green',s=100)
plt.plot(X,y_pred2, color='purple')
plt.show()
plt.scatter(X,y, c='red', s=100)
plt.plot(X,y_pred3, c='olive')
plt.show()

