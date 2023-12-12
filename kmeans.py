import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

#load the iris dataset
iris = pd.read_csv("/machine_learning/iris.csv")
x = iris[['sepal_length','sepal_width','petal_length','petal_width']]
y_label = iris['species']

#initialize a label encoder to convert string labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)

#create a k means model with a specified number of clusters, n_init describes the number of times algorithm will be run with different centroid initialization
model = KMeans(n_clusters=3,n_init=10,random_state=42)
model.fit(x)

#obtain the labels given by kmeans
labels = model.labels_
ari = adjusted_rand_score(y,labels) #to measure dissimilarity between the original y and predicted labels
print(f'Adjusted rand index:{ari}')

#visualize the original dataset
plt.scatter(x.values[:,0],x.values[:,1],c=y,cmap='viridis',edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original iris dataset')
plt.show()

#visualize kmeans clustering
plt.scatter(x.values[:,0],x.values[:,1],c=labels,cmap='viridis',edgecolors='k')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c='red',marker='x',s=220,label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means clustering on iris dataset')
plt.legend()
plt.show()