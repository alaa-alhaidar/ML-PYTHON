import  tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# documentations on https://www.tensorflow.org/
# The dataset consists of 60,000 training images and 10,000 test images,
# with each image being a 28x28 grayscale picture of a clothing item

data= keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = data.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

class_names=['T SHIRT', 'TROUSERS','PULLOVER','DRESS', 'COAT','SANDAL','SANDAL','SHIRT','SNEAKER','BAG','ANKLE']
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc : ", test_acc)

prediction = model.predict([test_images])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("prediction" + class_names[np.argmax(prediction[i])])
    plt.show()
