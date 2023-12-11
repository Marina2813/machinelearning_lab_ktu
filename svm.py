import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#replace datasets.csv with actual file path
iris = pd.read_csv("C:/Users/marin/OneDrive/Desktop/machine_learning/iris.csv")

#seperate feature x and target variable y
x = iris[['sepal_length','sepal_width','petal_length','petal_width']]
y_label = iris[['species']]

#convert string labels to numerical labels
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y_label)
print(x.head())
print(x.tail())
print("The dimension of dataset is",iris.shape)

#split the datasets into training and test datasets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=125)

#create instance of svm model using kernel for seperating linearly seperable data
model = SVC(kernel="linear")
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Model Predictions")
print(y_pred)
print("Accuracy")
print(accuracy_score(y_test,y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))