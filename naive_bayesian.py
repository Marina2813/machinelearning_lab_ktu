import pandas as pd
from sklearn import datasets

#load the iris dataset
iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y= iris.target
print(x.head())

#split the dataset into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=125)

#train the model 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

#predict using test dataset
predicted = model.predict(x_test)
print(predicted)

new_data = pd.DataFrame([[5.0,3.6,1.4,0.2]],columns=iris.feature_names)
p = model.predict(new_data)
print(p)

#to find accuracy
from sklearn.metrics import accuracy_score,precision_score,recall_score
print("Accuracy: ",accuracy_score(y_test,predicted))

precision = precision_score(y_test,predicted,average=None)
recall = recall_score(y_test, predicted,average=None)
print("Precision: ",precision)
print("recall: ",recall)