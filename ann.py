import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

#load the wine dataset
wine = pd.read_csv("/machine_learning/wine.csv")
x = wine[['Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline']]
y = wine['Wine']

#split the dataset into training and test datasets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#standardize the input features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#create an MLP Classifier model
model = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=1000,random_state=42)
model.fit(x_train,y_train)

#make predictions
y_pred = model.predict(x_test)

#print the confusion matrix
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))

#print the accuracy
acc = accuracy_score(y_test,y_pred)
print(acc)

# to use my own data to make predictions
new_data = pd.DataFrame({
    'Alcohol': [13.5],
    'Malic.acid': [2.5],
    'Ash': [2.6],
    'Acl': [20],
    'Mg': [100],
    'Phenols': [2.8],
    'Flavanoids': [3.0],
    'Nonflavanoid.phenols': [0.2],
    'Proanth': [1.5],
    'Color.int': [4.0],
    'Hue': [1.1],
    'OD': [2.8],
    'Proline': [500]
})

# Standardize the new data using the same scaler used for training
new_data_standardized = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_standardized)

# Display the predictions
print("Predictions:")
print(predictions)
