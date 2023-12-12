import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

#replace "/machine_learning/iris.csv" with your actual file path
df = pd.read_csv("machine_learning/iris.csv")

# Separate features (x) and target variable (y_label)
x = df.drop('species',axis=1)
y_label = df['species']

# Encode the categorical target variable using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)

# Standardize the features using StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# Apply PCA to reduce the dimensionality to 2 components
pca = PCA(n_components=2)
x_new = pca.fit_transform(x_std)

# Print the original and reduced number of features
print("Original number of features",x_std.shape[1])
print("reduced number of features",x_new.shape[1])


# Define colors for each class
colors = ['red','green','yellow']
for target,color in zip(np.unique(y),colors):
    plt.scatter(x_new[y==target,0],x_new[y==target,1],color=color,label=target)

# Set labels and title for the plot
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Custom Dataset')
plt.legend()

# Show the plot
plt.show()