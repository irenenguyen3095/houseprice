#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:47:55 2019

@author: Irene
"""
#dictionaries
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats

#import dataset
df = pd.read_csv('/Users/saturn/Downloads/kc_house_data.csv')
print(df.columns)
#X = df.drop(columns = ['price'])
#y = df['price']

print(df.describe())

#analysing target 
pd.options.display.float_format = '{:,.2f}'.format

df['price'].dtypes
print(df['price'].describe())

def checkDist(dist):
    sns.distplot(dist, color= 'pink')
    plt.title('Distribution')
    plt.legend(['Normal dist. ($Skew=$ {:.2f} and $Kurtosis=$ {:.2f} )'.format(dist.skew(), dist.kurt())],
            loc='best')
    fig = plt.figure()
    res = stats.probplot(dist, plot=plt)
    plt.show()
    return

print('Distribution of price:')
print(checkDist(df['price']))
print('This distribution is right skewed. To fix this, i will use log transform.')
print('Distribution of price with log transform:')
print(checkDist(np.log1p(df["price"])))

df['price']= np.log1p(df["price"])

#deal with missing datas:
print(df.isnull().sum())
#was für ein unrealisitsches Dataset

#data correlation
#heatmap for correlation
corr = df.corr()
plt.subplots(figsize=(10,7))
print(sns.heatmap(corr, vmax=0.8,square=True, cmap="YlGnBu"))

#a closer look to 10 variables most correlated to price
var = corr.nlargest(12, 'price')['price'].index
#var = var.drop(column='lat')
corrm = np.corrcoef(df[var].values.T)
#sns.set(font_scale=1.25)
plt.subplots(figsize=(10,7))
print(sns.heatmap(corrm, cbar=True, annot=True, square=True, cmap="YlGnBu", fmt='.2f', annot_kws={'size': 10}, yticklabels=var.values, xticklabels=var.values))

#scatterplot
sns.set()
sns.pairplot(df[var], size = 2.5)
plt.show();

for m in var:
    df_new= pd.concat([df[var]], axis=1)

df_new= df_new.drop(columns= ['lat','sqft_basement'])    


print(len(df_new['bathrooms'].unique()))

categorical= []
numerical =[]

for col in df_new:
    if len(df_new[col].unique())<31:
        categorical.append(col)
    else:
        numerical.append(col)
        
for feat in numerical:
    checkDist(df_new[feat])
    checkDist(np.log1p(df_new[feat]))
    
for a in ('price','sqft_living15','sqft_living', 'sqft_living'):
    df_new[a]= np.log1p(df_new[a])


len(df_new['bathrooms'].unique())
var_new=list(var)

#scatterplot for numerical features
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
cols =2
rows= 2
fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
for r in range(0,rows):
    for c in range(0,cols):  
        i = r*cols+c
        if i < len(var_new):
             sns.scatterplot(x=numerical[i], y='price', data=df, ax = axs[r][c], palette= 'Set2')
plt.tight_layout()    
plt.show()   

#boxplot for categorical features
cat_cols =3
cat_rows= 2
fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
for r in range(0,rows):
    for c in range(0,cols):  
        i = r*cols+c
        if i < len(var_new):
             sns.boxplot(x=categorical[i], y='price', data=df, ax = axs[r][c])
plt.tight_layout()    
plt.show()   

#deal with bathrooms features
print(df_new['bathrooms'].unique())

#price with grade
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
fig, ax = plt.subplots(figsize=(15,10))
sns.catplot(x='bathrooms', y='price',data= df_new, ax=ax, kind= 'boxen')


plt.scatter(x=df['price'], y= df['sqft_living15'])
plt.ylabel('Price', fontsize=13)
plt.xlabel('sqft_living15', fontsize=13)
plt.show()

X = df_new.drop(columns = ['price'])
y =df_new.iloc[:, 0].values

#creat train and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
# Train the model using the training sets
reg= LinearRegression(fit_intercept=True, normalize=False).fit(X_train,y_train)
# Make predictions using the testing set
y_pred = reg.predict(X_test)
print(y_pred.shape)
# The coefficients
from sklearn.metrics import mean_squared_error, r2_score
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train,y_train)
y_pred_ridge= ridge.predict(X_test)
print('Coefficients: \n', ridge.coef_)
print('Variance score: %.2f' % r2_score(y_test, y_pred_ridge))

