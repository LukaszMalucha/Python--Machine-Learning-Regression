# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Feature Scaling - needed for rbf SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()    ## create object for scaling X and y
sc_y = StandardScaler()
X= sc_X.fit_transform(X)   ## use those objects for scaling
y= sc_y.fit_transform(y)



# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')                   ## Kernel choice - linear, poly or gaussian(rbf)
regressor.fit(X,y)


# Predicting a new result - has to be changed as scaling was applied 
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))   ## transform value into array with numpy.array
## Two square brackests to create an array

## Inverse scale transformation to get real scale
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
## For smoother plot:
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## Level 10 is omitted as model considers it as an outlier