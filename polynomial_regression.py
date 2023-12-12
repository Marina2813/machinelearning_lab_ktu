import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Set the degree of the polynomial features
poly = PolynomialFeatures(degree= 2)

# Create input data (x) and corresponding output data (y)
x = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y = np.array([3,4,5,7,10,8,9,10,10,23]).reshape(-1,1)

# Plot the original data points
plt.scatter(x,y)
plt.title('Original Data Points')
plt.show()

# Transform the input data to include polynomial features
poly_x = poly.fit_transform(x)

# Create a Linear Regression model
R = LinearRegression()

# Fit the model to the polynomial features
R.fit(poly_x,y)

# Predict the output using the trained model
y_pred = R.predict(poly_x)

#plot the original data points and regression line
plt.scatter(x,y)
plt.plot(x,y_pred,c="red")
plt.show() 