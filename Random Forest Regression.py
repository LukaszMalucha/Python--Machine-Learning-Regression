## Random Forest Regression
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Random Forest Regression Model to the dataset (3 STEPS)
from sklearn.ensemble import RandomForestRegressor                      # import from library
regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)  # call the method, no need for criterion as we use default mse
regressor.fit(X, y)                                                     # fit regressor into dataset

# Predicting a new result
y_pred = regressor.predict(6.5)


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)                        ## increase the resolution
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Reality Check (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## Random Forest model prediction is most accurate one from all the regression models. 












