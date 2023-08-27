import tensorflow
import pandas as pd
import numpy as np
import keras
import sklearn
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.neighbors import KNeighborsClassifier

# KNN algorithm to classify points based on distance
data = pd.read_csv("car.data.csv")


# change categorical data to integer
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"
x= list(zip(buying, maint,persons,lug_boot,safety,cls))
y= list(cls)
# train the data to predict the class
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


model = KNeighborsClassifier(9)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted= model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("Predicted", names[predicted[x]], "Data", x_test[x], "actual: ",names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print(n)