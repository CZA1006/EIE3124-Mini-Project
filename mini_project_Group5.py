# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:41:27 2022

@author: Acer
"""

"""
Part 1  data preprocess
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import pearsonr

#import dataset
df = pd.read_csv('healthy_lifestyle_city_2021.csv') #raw data
#print(df)
#%%
#clear missing data
df=df.drop(df[(df['Sunshine hours(City)']=='-')|(df['Pollution(Index score) (City)']=='-')| #detect all the missing part and delete the whole row
          (df['Annual avg. hours worked']=='-')|(df['Sunshine hours(City)']=='-')].index)
#print(df)
#%%
#process raw data
happiness=df['Happiness levels(Country)']
sunshine=pd.to_numeric(df['Sunshine hours(City)'])   #change type from object to int/float
BoW=pd.to_numeric(df['Cost of a bottle of water(City)'].str.slice(1)) #strip the first character
obesity=pd.to_numeric(df['Obesity levels(Country)'].str.slice(0,-1))  #strip the last character
life=df['Life expectancy(years) (Country)']
pollution=pd.to_numeric(df['Pollution(Index score) (City)'])
work_hour=pd.to_numeric(df['Annual avg. hours worked'])
activity=pd.to_numeric(df['Outdoor activities(City)'])
take_out=pd.to_numeric(df['Number of take out places(City)'])
gym=pd.to_numeric(df['Cost of a monthly gym membership(City)'].str.slice(1))
factors=[sunshine,BoW,obesity,life,pollution,work_hour,activity,take_out,gym]
factors_name=['sunshine','BoW','obesity','life','pollution','work_hour','activity','take_out','gym']
#print(factors)

#%%
"""
Part 2  correlation coeficient
"""
print('---------Part 2 output-----------')
# calculate Correlation Coefficient
from scipy import stats
corr=[]
for i in factors:
    # corr1,_=pearsonr(i,happiness)
    # corr.append(corr1)
    corr.append(stats.pearsonr(i,happiness))
#print(corr)
print('Correlation Coefficient:\n"Attribute : (corr,p-value)"')
for i,j in zip(factors_name,corr):
    print(i.ljust(9),':',j)
#%%
#plots
plt.subplots_adjust(left=0,bottom=0,wspace=0.4,hspace=0.6)
for i in range (0,9):
    plt.subplot(3,3,i+1)         # show 9 plots in 3*3 form
    plt.scatter(factors[i],happiness,s=3)
    plt.xlabel(factors_name[i])
    plt.ylabel('happiness')

#%%
"""
Part 3  MLE
"""
print('\n---------Part 3 output-----------')
# Linear Regression
from sklearn import linear_model

regr = linear_model.LinearRegression()
#regr.fit(df[['Outdoor activities(City)']], happiness)
regr.fit(pd.concat(factors, axis=1), happiness)
predictedCO1 = regr.predict([[0]*9]) #w0
#print(predictedCO1)
#print(regr.coef_)

print('w0= ',predictedCO1)
print('w1-9= ',regr.coef_)
w=np.dstack((factors_name,regr.coef_))
print("   'Attribute'  'weighting'\n",w)

#%%
"""
Part 4  Module evaluate
"""
# Method evaluate
print('\n---------Part 4 output-----------')
data=pd.concat(factors, axis=1)
#print(data.shape)

#use front 20 row to train the module and others to evaluate
regr_E = linear_model.LinearRegression()
regr_E.fit(data[:20], happiness[:20])
w0=(regr_E.predict([[0]*9]))[0]

#store the predict data in pred
pred=[]
for i in range(0,data.shape[0]):
    pred.append((data.iloc[i]*regr_E.coef_).sum()+w0)
#print(pred)
#print(happiness)
#%%
#Standard error of prediction
err=pred-happiness   #each absolute error
Square_err=err**2    #each square error
T_se=np.sqrt(Square_err[:20].sum()/20)             #standard error for training set
E_se=np.sqrt(Square_err[20:].sum()/(len(err)-20))  #standard error for prediction set
print('T_se= ',T_se)
print('E_se= ',E_se)




#%%
pd.set_option('display.max_columns', None)







