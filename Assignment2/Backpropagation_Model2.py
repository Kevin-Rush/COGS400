import numpy as np
import tensorflow as tf
#import mnist 
#import matplotlib.plyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images

train_X = tf.keras.utils.normalize(train_X, axis=1)
test_X = tf.keras.utils.normalize(test_X, axis=1)

train_X = train_X.reshape((-1, 784))
test_X = test_X.reshape((-1, 784))

input_size = 784
hidden_size = 25 
output_size = 10

NN = Sequential()
NN.add(Dense(hidden_size, activation="sigmoid", input_dim=input_size))
NN.add(Dense(output_size, activation="sigmoid"))

NN.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
NN.fit(train_X, to_categorical(train_y), epochs = 10)

NN.fit(test_X, to_categorical(test_y), epochs = 1)

predictions = NN.predict(test_y[:5])
print(predictions)
"""
loss, accuracy = NN.evaluate(test_X, test_y)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
"""
"""
#Confution Matrix and Classification Report
Y_pred = NN.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print("Confusion Matrix: ")
print(confusion_matrix(validation_generator.classes, y_pred))
print("Classification Report")

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
"""