
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("student-mat.csv", sep=";")
feature_names = data.columns
'''
school
sex
age
address
famsize
Pstatus
Medu
Fedu
Mjob
Fjob
reason
guardian
traveltime
studytime
failures
schoolsup
famsup
paid
activities
nursery
higher
internet
romantic
famrel
freetime
goout
Dalc
Walc
health
absences
G1
G2
G3
'''
print("Feature names:")
for feature in feature_names:
    print(feature)

# filter the data to have some of the column , not all of them
#data = data[["G1", "G2", "G3", "sex", "studytime", "failures", "absences"]]

print(data["G1"])

# Test-train split supervised learning
'''
data["G1"]: This extracts the values from the "G1" column of the data DataFrame. 
 This will be our feature or input data (X).

data["G3"]: This extracts the values from the "G3" column of the data DataFrame. 
 This will be our target or output data (y).
'''
x_train, x_test, y_train, y_test = train_test_split(data["G1"], data["G3"])
'''
# reshape(-1, 1): The .reshape() method is used to change the shape of the array. In this case, (-1, 1) is passed as the argument to reshape(), 
# which means that we want to reshape the array to have one column and as many rows as necessary to accommodate the data.

'''

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

'''
plt.scatter(x_train, y_train, label="Training data", color="r", alpha=0.7)
plt.scatter(x_test, y_test, label="Test data", color="g", alpha=0.7)
plt.xlabel("G1")
plt.ylabel("G3")
plt.title("Test Train Split")
plt.legend()
plt.show()
'''


linear = LinearRegression()

# train the model labeled
linear.fit(x_train, y_train)

# save the trained model
with open("linearModel1.pickel","wb")as f:
    pickle.dump(linear,f)

'''
The predict method in a machine learning model is used to make predictions on new, 
unseen data (X) based on the learned patterns from the training data.
'''
# restore the saved model
pickle_in = open("linearModel1.pickel", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

pre=linear.predict(np.array([[15]]))[0]
print("Predicted G3 value for G1=15:", pre)


score=linear.score(x_test, y_test)
print(score)

plt.scatter(x_train, y_train, label="Training data", color="b", alpha=0.7)
plt.scatter(x_test, y_test, label="Test data", color="g", alpha=0.7)
plt.plot(x_test, predictions, color="b", label="Linear Regression", linewidth=1)
plt.xlabel("G1")
plt.ylabel("G3")
plt.title("Test Train Split with Linear Regression")
plt.legend()
plt.show()

# Select three columns for the 3D plot
x = data["G1"]
y = data["G2"]
z = data["G3"]

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z, c='r', marker='o')

# Set labels for the axes
ax.set_xlabel("G1")
ax.set_ylabel("G2")
ax.set_zlabel("G3")

# Show the 3D plot
plt.show()












