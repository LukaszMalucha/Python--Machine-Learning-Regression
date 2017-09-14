## Polynomial Regression (with comparison to linear regression)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values  ## Skip position name column as redundant to 'level' column (10 = CEO)
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
## Small amount of information - we can't split the dataset 
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

## Fitting Linear Regression to the dataset (for comparison)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

## Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)           ## Transform X into new matrix composed with X^2
X_poly = poly_reg.fit_transform(X)                     ## First fit, then transform the object into poly - used in a plot later
## Three colums are present - column of ones as regression constant b0, x1 and x^2


## Create object to fit poly into X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

## Visualising The Linear Regression results
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')       ## fit linear prediction inot the plot
plt.title('Reality Check (Linear Regression)')
plt.xlabel('Position Level')
plt.ylevel('Salary')
plt.show()                                           ## DON'T FORGET TO SHOW THE GRAPH 

          
## Visualising The Polynomial Regression results as Linear Regression doesn't fit.
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')       ## Swap two objects - lin_reg_2 as well as matrix with polynomial terms
plt.title('Reality Check (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylevel('Salary')
plt.show()                

###Polynomial model fits much better, but let's swap polynomial degree to '4'                      

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)          
X_poly = poly_reg.fit_transform(X)                     
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

### Best fitting model, but let's create continous curve
X_grid = np.arange(min(X), max(X), 0.1)   ## First create the vector
X_grid = X_grid.reshape(len(X_grid),1)    ## Now, turn it into matrix      
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')  ## Swap to X_gridin both places   
plt.title('Reality Check (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylevel('Salary')
plt.show()                

## Predicting a new result with Linear Regression
lin_reg.predict(6.5)    ## Level
## Bad prediction

## Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

## Accurate prediction
















