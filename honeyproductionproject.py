# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:05:23 2021
As you may have already heard, the honeybees are in a precarious state right now. 
I may have seen articles about the decline of the honeybee population for various reasons. 
In this project want to investigate this decline and how the trends of the past predict the future for the honeybees.
The data is in a DataFrame for you about honey production in the United States from Kaggle. 
Data of the honey production in U.S. provided in my github repository as a csv file.
@author: Sarp
"""

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/DELL/.spyder-py3/honeyproduction.csv")
print(df.head())
#Total production of honey per year. 
#Whatever we used group by will be the X axis in our regression model
#The corresponding values(total production in this case) will be the y axis of the regression model.
prod_per_year=df.groupby('year').totalprod.mean().reset_index()
#Create a variable called X that is the column of years in this prod_per_year DataFrame.
X=prod_per_year['year']
X=X.values.reshape(-1,1)
#Create a variable called y that is the totalprod column in the prod_per_year dataset.
y=prod_per_year['totalprod']
#Using plt.scatter(), plot y vs X as a scatterplot.
plt.scatter(X,y)
#Create a linear regression model from scikit-learn and call it regr.
regr=linear_model.LinearRegression()
#Fit the model to the data by using .fit(). You can feed X into your regr model by passing it in as a parameter of .fit().
regr.fit(X,y)
#Print out the slope of the line (stored in a list called regr.coef_) and the intercept of the line (regr.intercept_).
print(regr.coef_[0]) #this is our m(slope)
print(regr.intercept_)#this is our b(intercept)
#Create a list called y_predict that is the predictions your regr model would make on the X data.
y_predict=regr.predict(X)
#Plot y_predict vs X as a line, on top of your scatterplot using plt.plot().
plt.plot(X,y_predict)
plt.show()
#So, it looks like the production of honey has been in decline, according to this linear model. 
#Let’s predict what the year 2050 may look like in terms of honey production.
#Let’s create a NumPy array called X_future that is the range from 2013 to 2050. 
X_future=np.array(range(2013,2051))
#After creating that array, we need to reshape it for scikit-learn.
#You can think of reshape() as rotating this array. Rather than one big row of numbers
#X_future is now a big column of numbers — there’s one number in each row.
X_future=X_future.reshape(-1,1)
#Create a list called future_predict that is the y-values that your regr model would predict for the values of X_future.
future_predict=regr.predict(X_future)   
plt.plot(X_future,future_predict)
plt.show()