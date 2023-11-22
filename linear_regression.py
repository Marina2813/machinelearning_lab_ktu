from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv("C:/Users/marin/OneDrive/Desktop/machine_learning/student_scores.csv")
df_binary= df[['Hours','Scores']]
print(df_binary.head())

x = np.array(df_binary["Hours"]).reshape(-1,1)
y = np.array(df_binary["Scores"]).reshape(-1,1)

plt.scatter(x,y,color="green")
plt.show()

R=LinearRegression()
R.fit(x,y)

y_pred=R.predict(x)
print(y_pred[1:5])
print(f'intercept ={R.intercept_}')
print(f'coeff ={R.coef_}')
print(f'Error = {mean_squared_error(y,y_pred)}')

plt.plot(x,y_pred,color="green")
plt.scatter(x,y)
plt.show()