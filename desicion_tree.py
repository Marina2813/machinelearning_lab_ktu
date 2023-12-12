import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

#replace the file path with your actual file path 
df = pd.read_csv("/machine_learning/iris.csv")
x = df.drop('species',axis=1)
y_label = df['species']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)

# Split the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.33)

# Create a Decision Tree model with specified parameters
model = DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3)
dtree = model.fit(x_train,y_train)

# Export the decision tree visualization to a DOT file
tree.export_graphviz(dtree,out_file='Dtree.dot')

# Read the DOT file and create a graph visualization
with open("Dtree.dot") as dot_file:
    dot_data = dot_file.read()


# Render and save the decision tree graph as an image
graph = graphviz.Source(dot_data)
graph.render("tree", format="png", cleanup=True)

#predictions on test  dataset
y_pred = dtree.predict(x_test)
print(y_pred)

#prediction on a new data point
p = dtree.predict([[4.5,5.6,6.5,2.3]])
print(p)