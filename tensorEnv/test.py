import tensorflow
import pandas as pd
import numpy as np
import keras
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"
x = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
 # the model were saved so the code can be archived
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) # train the data
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc > best:
        best = acc
        # save the trained model
        with open("studentmodel.pickel","wb")as f:
            pickle.dump(linear,f)
            
print("the best result so far: ",best)
'''

# restore the saved model
pickle_in = open("studentmodel.pickel", "rb")
linear = pickle.load(pickle_in)

print('coefficient: \n', linear.coef_)  # 5 variables predictors
print('intercept: \n', linear.intercept_)
# predict the value of the dependent variable G3 based on trained data
predictions = linear.predict(x_test)

# print the result to compare
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
