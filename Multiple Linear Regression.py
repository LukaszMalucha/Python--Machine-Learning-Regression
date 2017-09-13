# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])   ## '3' corresponds to 'State' variable which need to be replaced 
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable - No need as dependent variable is not categorical
"""labelencoder_y = LabelEncoder()       
y = labelencoder_y.fit_transform(y)"""

### AVOIDING THE DUMMY VARIABLE TRAP  !!!!!  ###
X = X[:,1:]   ## removing first column of 'X'


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)  

# Compare y_pred results with y_test....

## Building better model using Backward Elimination
import statsmodels.formula.api as sm

# Add column of '1' as 'x0' for b0x0 constant - statsmodel doesn't do that automatically. 
'X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)' ## (50,1) stands for shape -50 rows and one column
### TRICK***  - Invert values to add it as a first column, not a last one
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) ## add X to column of '1' instead (np - NumPy)

### BACKWARD ELIMINATION START###

# Create matrix of independables with high impact

X_opt = X[:,[0,1,2,3,4,5]]    ## Matrix with all the colums separated for elimination
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()  ## Intercept not included by default - that's why column on ones was created

### FIND P-value for each independent variable (treshold < 6%)

regressor_OLS.summary()  ## Table of statistical values

## Table shows that variable x2 has the highest value and has to be eliminated

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## Table shows that variable x1 has the highest value and has to be eliminated

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## Table shows that variable x4 has the highest value and has to be eliminated

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## x3(New York) p-value is very low while x5(marketing) has p-value = 6%
## Both variables have significant predict power on profit





