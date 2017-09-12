# Simple Linear Regression

# Simple Linear Regression with sklearn

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('salary.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) ## 1/3 for size of 10

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

## Fitting Simple Linear Regression to the Training set (with sklearn)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()         
regressor.fit(X_train,y_train)  ## fir regression into x_train model

## Predicting Test set results

y_pred = regressor.predict(X_test)     ## vector of prediction of dependent variable

## Have a look and compare y-test with y_pred

## Visualising Training set results (with matplotlib) 

plt.scatter(X_train, y_train, color = 'red') ## first place the employees on the plot
plt.plot(X_train, regressor.predict(X_train), color = 'blue') ## compare with predicted salaries
plt.title('Salary vs Experience(Training set)')      
plt.xlabel('Years of experience')
plt.ylabel('Salary')            
plt.show()        ## End of the graph - ready to show it          


## Visualising Test set results (with matplotlib) 
plt.scatter(X_test, y_test, color = 'red') ## first place test set employees on the plot
plt.plot(X_train, regressor.predict(X_train), color = 'blue') ## exactly same regression line as previosly
plt.title('Salary vs Experience(Test set)')      
plt.xlabel('Years of experience')
plt.ylabel('Salary')            
plt.show()        ## End of the graph - ready to show it  
