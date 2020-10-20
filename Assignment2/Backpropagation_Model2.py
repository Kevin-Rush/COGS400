import tensorflow as tf
from tensorflow import keras
import numpy as np

(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images
    
train_X = tf.keras.utils.normalize(train_X, axis=1)
test_X = tf.keras.utils.normalize(test_X, axis=1)

input_size = 784
hidden_size = 100
output_size = 1

NN = keras.Sequential()
NN.add(tf.keras.layers.Dense(hidden_size, input_dim=input_size, activation="sigmoid"))
NN.add(tf.keras.layers.Dense(output_size, activation="sigmoid"))

NN.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1]*train_X.shape[2])
train_y = tf.one_hot(train_y, 60000, on_value=1.0, off_value=0.0,)

NN.fit(train_X, train_y, epochs=20)

loss, accuracy = NN.evaluate(test_X, test_y)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
"""
#Confution Matrix and Classification Report
Y_pred = NN.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print("Confusion Matrix: ")
print(confusion_matrix(validation_generator.classes, y_pred))
print("Classification Report")

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
"""