import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree= 2)
x = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y = np.array([3,4,5,7,10,8,9,10,10,23]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

poly_x = poly.fit_transform(x)
R = LinearRegression()
R.fit(poly_x,y)

y_pred = R.predict(poly_x)
plt.scatter(x,y)
plt.plot(x,y_pred,c="red")
plt.show()