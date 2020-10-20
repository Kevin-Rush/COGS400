"""
Back propagation - Assignment 2

Kevin Rush
10052650
11kr28
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class BackpropNN():
    def __init__(self):
        #parameters
        self.input_size = 784
        self.hidden_size = 50
        self.output_size = 10
        self.learning_rate = 0.5
        self.matrix = np.zeros((10, 10))

        #weights
        self.weights_input_hidden = 2*np.random.random((self.input_size, self.hidden_size)) - 1 #784x7 weight matrix from input to hidden layer
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size)) - 1 #7x10 weight matrix from hidden to output layer

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))
    
    def sigmoid_derivaitve (self, input):
        #print(input)
        return input * (1 - input)

    def find_index_largest(self, array):
        largest = -1
        index = -1
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i
        
        return index

    def one_hot_encoding(self, array):
        one_array = np.zeros(10)
        largest = -1
        index = -1
        for i in range(len(array[0])):
            if largest < array[0][i]:
                largest = array[0][i]
                index = i

        one_array[index] = 1

        return one_array

    def feedforward (self, X):                                  #forward propogation through the network
        self.z = np.dot(X.T, self.weights_input_hidden)           #dot product of input matrix and first set of weights, returns 784x7 matrix
        self.z2 = self.sigmoid(self.z)                          #activation function, returns the z matrix through the sigmoid function
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   #dot product of hidden layer (z2) and second set of weights, returns 784x10 matrix
        output = self.sigmoid(self.z3)
        #self.system_check()    
        return output

    def backpropogation(self, X, y, output):                                    #backward propogate through the network
        
        output_error = (y - output)                                          #error in output
        output_delta = self.learning_rate*output_error*self.sigmoid_derivaitve(output)*self.z3 
        """
        print("output error: "+str(output_error))
        print("output delta: "+str(output_delta))
"""
        z2_error = output_delta.dot(self.weights_hidden_output.T)      #z2_error: how much hidden layer weights contribute to output error
        z2_delta = self.learning_rate*z2_error*self.sigmoid_derivaitve(self.z2)*self.z          #applying the derivative of sigmoid to z2 error
        """
        print("Z2's:")
        print(z2_error)
        print(z2_delta)
        print("weights:")
        print(self.weights_input_hidden[0])
        print(self.weights_hidden_output[0])
        print("X")
        for i in range(20):
            print(X[i])
        print("dot")
        print(np.dot(X, z2_delta))
"""
        self.weights_input_hidden += z2_delta                     #adjusting first set (input --> hidden) weights
        self.weights_hidden_output += self.z2.T.dot(output_delta)          #ajdusting the second set (hidden --> output) weights
        """
        print(self.weights_input_hidden[0])
        print(self.weights_hidden_output[0])
        """

    def train(self, X, y):
        array = np.zeros((len(y), 10))
        for i in range(len(y)):
            array[i][y[i]] = 1

        for j in range(10):                                  #iteratior
            for i in range(len(X)):
                image = X[i].reshape(784,1)
                #print(image)
                output = self.feedforward(image)             #sending one 28x28 matrix that represents an image
                self.backpropogation(image, array[i], output)    #y is a single digit               
    
    def test(self, X, y):
        array = np.zeros((len(y), 10))
        for i in range(len(y)):
            array[i][y[i]] = 1
        
        correct = 0
        incorrect = 0
        for i in range(len(X)):
            array = np.zeros((len(y), 10))
            for j in range(len(y)):
                array[j][y[j]] = 1
            image = X[i].reshape(784,1)
            output = self.feedforward(image)
            output = self.one_hot_encoding(output)
            index_output = self.find_index_largest(output)
            if index_output == self.find_index_largest(array[i]):
                correct += 1
                self.matrix[index_output][index_output] += 1
            else:
                incorrect += 1

        return correct, incorrect

if __name__ == "__main__":      #main function to run program and generate output for display purposes
    NN = BackpropNN()

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images
    
    train_X = tf.keras.utils.normalize(train_X, axis=1)
    test_X = tf.keras.utils.normalize(test_X, axis=1)
    
    print(NN.weights_input_hidden[0])
    print(NN.weights_hidden_output[0])
    
    NN.train(train_X, train_y)

    print(NN.weights_input_hidden[0])
    print(NN.weights_hidden_output[0])

    print(NN.test(test_X, test_y))
    #print(NN.matrix)

