import sklearn
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Documentation
# https://scikit-learn.org/stable/modules/svm.html

'''
In this code, we are performing a binary classification task using the Support Vector Machine (SVM) 
algorithm to predict the class of breast cancer tumors as either malignant (1) or benign (0).
x = cancer.data: x contains the features or input data of the breast cancer dataset. 
It represents the characteristics of tumors, such as mean radius, mean texture, etc.

y = cancer.target: y contains the target or output data, 
which represents the class labels of the tumors (1 for malignant and 0 for benign).
'''
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)


x = cancer.data
y = cancer.target


num_features = x.shape[1]
print("Number of features in x:", num_features)

# features of each point x, every x contains the following attribute
# mean radius
# mean texture
# mean perimeter
# mean area
# ...
feature_names = cancer.feature_names
print("Feature names:")
for feature in feature_names:
    print(feature)

# train the data to predict the class. 90 % used to train this model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

classes = ["malignant", "benign"]

clf = svm.SVC(kernel="poly", C=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = sklearn.metrics.accuracy_score(y_test, y_pred)
print("accuracy this model ",acc)

# Select three columns for the 3D plot
x_axis = cancer.data[:, 0]
y_axis = cancer.data[:, 1]
z_axis = cancer.data[:, 2]
# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the 3D scatter plot for class 0 (malignant)
ax.scatter(x_axis[cancer.target == 0], y_axis[cancer.target == 0], z_axis[cancer.target == 0], c='r', marker='o', label='malignant')
# Plot the 3D scatter plot for class 1 (benign)
ax.scatter(x_axis[cancer.target == 1], y_axis[cancer.target == 1], z_axis[cancer.target == 1], c='b', marker='o', label='benign')

# Set labels for the axes
ax.set_xlabel("mean radius")
ax.set_ylabel("mean texture")
ax.set_zlabel("mean perimeter")

# Add a legend with two lines and their respective colors
ax.legend(loc='upper left', title='3D based on classes malignant, benign', labels=['malignant', 'benign'], frameon=False)

# Show the 3D plot
plt.show()

feat_x = 1
plt.scatter(x[:, 0], x[:, feat_x], c=y, cmap=plt.cm.Paired, s=10)
plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[feat_x])
plt.title("Scatter Plot of Breast Cancer Data")
plt.colorbar(label="Class Label")
plt.legend(loc='upper right')
plt.xlim(0, 30)
plt.ylim(0, 40)
plt.show()

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap=plt.cm.Paired, s=10)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

for i, class_name in enumerate(classes):
    plt.scatter([], [], c=plt.cm.Paired(i / 1.0), label=class_name)

plt.legend()
plt.xlim(0, 30)
plt.ylim(0, 40)
plt.show()








