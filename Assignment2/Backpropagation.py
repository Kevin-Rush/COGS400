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
    def __init__(self): #file_input_hidden, file_hidden_output):
        #parameters
        self.input_size = 784
        self.hidden_size = 25
        self.output_size = 10
        self.learning_rate = 0.5
        self.a = 0.05
        self.bias = 1
        self.matrix = np.zeros((10, 10))
        self.count = 0
        self.MSE = 9999
        self.avrg_MSE = 0
        self.iterations = 5

        self.weights_input_hidden_last_weight_change = np.zeros((self.input_size+1,self.hidden_size), dtype=float)
        self.weights_hidden_output_last_weight_change = np.zeros((self.hidden_size, self.output_size), dtype=float)

        self.weights_input_hidden = 2*np.random.random((self.input_size+1, self.hidden_size))-1 
        self.weights_hidden_output = 2*np.random.random((self.hidden_size, self.output_size))-1 


    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))
    
    def sigmoid_derivaitve (self, input):
        return input * (1 - input)

    def find_index_largest(self, array):
        largest = -1
        index = -1
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i
        
        return index

    def _actual_pred(self, array):
        one_array = np.zeros(10)
        largest = -1
        index = -1
        for i in range(len(array)):
            if largest < array[i]:
                largest = array[i]
                index = i

        one_array[index] = 1

        return one_array
    
    def loss_fn(self, y, output):
        mse = (y - output)**2
        actual_mse = 0
        for i in range(len(mse)):
            actual_mse += mse[i]
        
        self.MSE = actual_mse/len(mse)
        self.avrg_MSE = (self.avrg_MSE+self.MSE)/self.count 

    def feedforward (self, X):                                  
        
        self.z = np.dot(X.T, self.weights_input_hidden)         
        self.z2 = self.sigmoid(self.z)                          
        
        self.z3 = np.dot(self.z2, self.weights_hidden_output)   
        output = self.sigmoid(self.z3)

        return output

    def backpropogation(self, X, y, output):                          #backward propogate through the network
        
        self.loss_fn(y, output)
        self.output_error = (y-output)
        output_sig_der = self.sigmoid_derivaitve(output)
        output_change = self.output_error*output_sig_der
        output_delta = np.outer(output_change, self.z2) 

        error_matrix = self.weights_hidden_output*output_change
        error_matrix = np.sum(error_matrix, axis=1)

        hidden_sig_der = self.sigmoid_derivaitve(self.z2)
        errors_multiplied = error_matrix*hidden_sig_der
        z2_delta = np.outer(errors_multiplied, X)

        self.weights_hidden_output += self.learning_rate*output_delta.T + self.a*self.weights_hidden_output_last_weight_change  
        self.weights_input_hidden += self.learning_rate*z2_delta.T + self.a*self.weights_input_hidden_last_weight_change

        self.weights_hidden_output_last_weight_change = output_delta.T + self.a*self.weights_hidden_output_last_weight_change
        self.weights_input_hidden_last_weight_change = z2_delta.T + self.a*self.weights_input_hidden_last_weight_change  

    def train(self, X, y):
        array = np.zeros((len(y), self.output_size))
        for i in range(len(y)):
            array[i][y[i]] = 1

        j = 0
        for j in range(self.iterations)                                  #iterator
            for i in range(len(X)):
                image = X[i].reshape(self.input_size,1)
                image = np.append(image, [[self.bias]])
                self.count += 1
                output = self.feedforward(image)             #sending one 28x28 matrix that represents an image
                self.backpropogation(image, array[i], output)    #y is a single digit   

            j += 1
            print("J: "+str(j))
            print("MSE: "+str(self.MSE))  
    
    def test(self, X, y):

        correct = 0
        incorrect = 0
        for i in range(len(X)):
            array = np.zeros((len(y), self.output_size))              #one-hot encoding the actual output
            for j in range(len(y)):
                array[j][y[j]] = 1
            image = X[i].reshape(self.input_size,1)
            image = np.append(image, [[self.bias]])             #add in the bias
            output = self.feedforward(image)
            output = self._actual_pred(output)           #get an exact number as output
            index_output = self.find_index_largest(output)
            if index_output == self.find_index_largest(array[i]):   #compare if the index's match up
                correct += 1
            else:
                incorrect += 1

        return correct, incorrect

if __name__ == "__main__":      #main function to run program and generate output for display purposes
    
    NN = BackpropNN()

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data() #returns an array of 28 x 28 images
    
    train_X = tf.keras.utils.normalize(train_X, axis=1)
    test_X = tf.keras.utils.normalize(test_X, axis=1)
    
    tempIH = NN.weights_input_hidden.copy()
    tempHO = NN.weights_hidden_output.copy()

    NN.train(train_X, train_y)

    print(NN.test(test_X, test_y))
