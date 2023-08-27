
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("Heart_Attack.csv", sep=",")
feature_names = data.columns

print("Feature names:")
for feature in feature_names:
    print(feature)

# filter the data to have some of the column , not all of them
#data = data[["G1", "G2", "G3", "sex", "studytime", "failures", "absences"]]
mapping = {'negative': 0, 'positive': 1}

# Use the apply method with a lambda function to apply the mapping
data['class'] = data['class'].apply(lambda x: mapping[x])

predictor_var = "glucose"
predicted_var = "pressurehight"

print(data)

# Test-train split supervised learning

x_train, x_test, y_train, y_test = train_test_split(data[predictor_var], data[predicted_var], test_size=0.1)

'''
# reshape(-1, 1): The .reshape() method is used to change the shape of the array. In this case, (-1, 1) is passed as the argument to reshape(), 
# which means that we want to reshape the array to have one column and as many rows as necessary to accommodate the data.

'''

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

linear = LinearRegression()
# train the model labeled
linear.fit(x_train, y_train)


'''
The predict method in a machine learning model is used to make predictions on new, 
unseen data (X) based on the learned patterns from the training data.
'''

predictions = linear.predict(x_test)

score=linear.score(x_test, y_test)
print(score)

plt.scatter(x_train, y_train, label="Training data", color="b", alpha=0.7)
plt.scatter(x_test, y_test, label="Test data", color="g", alpha=0.7)
plt.plot(x_test, predictions, color="r", label="Linear Regression", linewidth=1)
plt.xlabel(predictor_var)
plt.ylabel(predicted_var)
plt.title("Test Train Split with Linear Regression")
plt.legend()
plt.show()



# Assuming 'data' is your DataFrame containing the columns 'pressurehight', 'glucose', and 'class'

# Create separate DataFrames for each class
class_0 = data[data['class'] == 0]
class_1 = data[data['class'] == 1]

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plots for each class separately
ax.scatter(class_0['pressurehight'], class_0['glucose'], class_0['class'], c='g', marker='o', label='Heart attack negativ')
ax.scatter(class_1['pressurehight'], class_1['glucose'], class_1['class'], c='r', marker='o', label='Heart attack positiv')

# Set labels for the axes
ax.set_xlabel("pressurehight")
ax.set_ylabel("glucose")
ax.set_zlabel("class")

# Add the legend
ax.legend()

# Show the 3D plot
plt.show()










