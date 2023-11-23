import numpy as np
from sklearn import datasets

wine = datasets.load_iris()
x = wine.data
y = wine.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=125)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=1000,random_state=42)

mlp.fit(x_train,y_train)
y_pred = mlp.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy = accuracy_score(y_pred,y_test)
print("Accuracy Score: ",accuracy)

cm = confusion_matrix(y_pred,y_test)
print("Confusion Matirx: ",cm)