from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#reading dataset from a csv file
df = pd.read_csv("C:/Users/marin/OneDrive/Desktop/machine_learning/dataset2.csv")

# selecting independent variables(x) and dependent variable(y)
x = df[["SqFt","Bedrooms"]]
y = df["Price"]
print(x.head())
print(y.head())

# Creating a Linear Regression model and fitting it to the data
R = LinearRegression()
R.fit(x,y)

#making predictions using the data
y_pred=R.predict(x)
print(y_pred[1:5])
print(f'intercept ={R.intercept_}')
print(f'coeff ={R.coef_}')
print(f'Error = {mean_squared_error(y,y_pred)}')

# Creating new data for prediction
new_data={'SqFt':[2180],'Bedrooms':[4]}
df = pd.DataFrame(new_data)
predicted_price = R.predict(df)
print(f"Predicted price={predicted_price}")